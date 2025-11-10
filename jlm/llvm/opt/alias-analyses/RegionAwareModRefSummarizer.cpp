/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/trace.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/TarjanScc.hpp>
#include <jlm/util/Worklist.hpp>

#include <map>
#include <queue>
#include <set>
#include <vector>

namespace jlm::llvm::aa
{

/**
 * allocas that are not defined in f(), and not defined in a predecessors of f() in the call graph,
 * can not be live inside f(). They are added to the DeadAllocaBlocklist.
 */
static const bool ENABLE_DEAD_ALLOCA_BLOCKLIST = !std::getenv("JLM_DISABLE_DEAD_ALLOCA_BLOCKLIST");

/**
 * In a region with an alloca definition, the MemoryNode representing the alloca does not need to
 * be routed into the region if the alloca is shown to be non-reentrant.
 * Such allocas are added to the NonReentrantBlocklist.
 */
static const bool ENABLE_NON_REENTRANT_ALLOCA_BLOCKLIST =
    !std::getenv("JLM_DISABLE_NON_REENTRANT_ALLOCA_BLOCKLIST");

/**
 * Operations like loads and stores have a size.
 * If the size is larger than the size of a memory represented by a memory node X,
 * X can be excluded from the Mod/Ref summary of the operation.
 */
static const bool ENABLE_OPERATION_SIZE_BLOCKING =
    !std::getenv("JLM_DISABLE_OPERATION_SIZE_BLOCKING");

/**
 * Constant memory, such as functions, constant globals and constant import, can never change.
 * We therefore never need to route their memory states through anything.
 */
static const bool ENABLE_CONSTANT_MEMORY_BLOCKING =
    !std::getenv("JLM_DISABLE_CONSTANT_MEMORY_BLOCKING");

/**
 * When propagating explicit loads and stores, should a memory node be skipped if
 * the target ModRefNode is flagged as loading/storing from external,
 * and the memory node is externally available, and large enough to be covered by the flag.
 */
static const bool ENABLE_MOD_REF_PIP = !std::getenv("JLM_DISABLE_MOD_REF_PIP");

/** \brief Region-aware mod/ref summarizer statistics
 *
 * The statistics collected when running the region-aware mod/ref summarizer.
 *
 * @see RegionAwareModRefSummarizer
 */
class RegionAwareModRefSummarizer::Statistics final : public util::Statistics
{
  static constexpr auto NumRvsdgRegionsLabel_ = "#RvsdgRegions";
  static constexpr auto NumSimpleAllocas_ = "#SimpleAllocas";
  static constexpr auto NumNonReentrantAllocas_ = "#NonReentrantAllocas";
  static constexpr auto NumCallGraphSccs_ = "#CallGraphSccs";

  static constexpr auto CallGraphTimer_ = "CallGraphTimer";
  static constexpr auto AllocasDeadInSccsTimer_ = "AllocasDeadInSccsTimer";
  static constexpr auto SimpleAllocasSetTimer_ = "SimpleAllocasSetTimer";
  static constexpr auto NonReentrantAllocaSetsTimer_ = "NonReentrantAllocaSetsTimer";
  static constexpr auto AnnotationTimer_ = "AnnotationTimer";
  static constexpr auto CreateExternalModRefNodeTimer_ = "CreateExternalModRefNodeTimer";
  static constexpr auto SolvingTimer_ = "SolvingTimer";
  static constexpr auto CreateMemoryNodeOrderingTimer_ = "CreateMemoryNodeOrderingTimer";
  static constexpr auto CreateModRefSummaryTimer_ = "CreateModRefSummaryTimer";

public:
  ~Statistics() override = default;

  explicit Statistics(const rvsdg::RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph)
      : util::Statistics(Id::RegionAwareModRefSummarizer, rvsdgModule.SourceFilePath().value())
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(&rvsdgModule.Rvsdg().GetRootRegion()));
    AddMeasurement(
        NumRvsdgRegionsLabel_,
        rvsdg::Region::NumRegions(rvsdgModule.Rvsdg().GetRootRegion()));
    AddMeasurement(Label::NumPointsToGraphMemoryNodes, pointsToGraph.numMemoryNodes());
  }

  void
  StartCallGraphStatistics()
  {
    AddTimer(CallGraphTimer_).start();
  }

  void
  StopCallGraphStatistics(size_t numSccs)
  {
    GetTimer(CallGraphTimer_).stop();
    AddMeasurement(NumCallGraphSccs_, numSccs);
  }

  void
  StartAllocasDeadInSccStatistics()
  {
    AddTimer(AllocasDeadInSccsTimer_).start();
  }

  void
  StopAllocasDeadInSccStatistics()
  {
    GetTimer(AllocasDeadInSccsTimer_).stop();
  }

  void
  StartCreateSimpleAllocasSetStatistics()
  {
    AddTimer(SimpleAllocasSetTimer_).start();
  }

  void
  StopCreateSimpleAllocasSetStatistics(uint64_t numSimpleAllocas)
  {
    GetTimer(SimpleAllocasSetTimer_).stop();
    AddMeasurement(NumSimpleAllocas_, numSimpleAllocas);
  }

  void
  StartCreateNonReentrantAllocaSetsStatistics()
  {
    AddTimer(NonReentrantAllocaSetsTimer_).start();
  }

  void
  StopCreateNonReentrantAllocaSetsStatistics(size_t numNonReentrantAllocas)
  {
    AddMeasurement(NumNonReentrantAllocas_, numNonReentrantAllocas);
    GetTimer(NonReentrantAllocaSetsTimer_).stop();
  }

  void
  StartCreateExternalModRefNode()
  {
    AddTimer(CreateExternalModRefNodeTimer_).start();
  }

  void
  StopCreateExternalModRefNode()
  {
    GetTimer(CreateExternalModRefNodeTimer_).stop();
  }

  void
  StartAnnotationStatistics()
  {
    AddTimer(AnnotationTimer_).start();
  }

  void
  StopAnnotationStatistics()
  {
    GetTimer(AnnotationTimer_).stop();
  }

  void
  StartSolvingStatistics()
  {
    AddTimer(SolvingTimer_).start();
  }

  void
  StopSolvingStatistics()
  {
    GetTimer(SolvingTimer_).stop();
  }

  void
  StartCreateMemoryNodeOrderingStatistics()
  {
    AddTimer(CreateMemoryNodeOrderingTimer_).start();
  }

  void
  StopCreateMemoryNodeOrderingStatistics()
  {
    GetTimer(CreateMemoryNodeOrderingTimer_).stop();
  }

  void
  StartCreateModRefSummaryStatistics()
  {
    AddTimer(CreateModRefSummaryTimer_).start();
  }

  void
  StopCreateModRefSummaryStatistics()
  {
    GetTimer(CreateModRefSummaryTimer_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const rvsdg::RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Statistics>(rvsdgModule, pointsToGraph);
  }
};

/**
 * Struct representing a node in the ModRefGraph.
 * Each node represents a set of PointsToGraph nodes that may be modified or referenced
 * by some operation, or within some region.
 */
struct RegionAwareModRefSummarizer::ModRefNode final
{
  // Constant indicating that the loadFromExternal and/or storeToExternal flags are not set.
  static constexpr uint32_t NotFlagged = (1 << 15) - 1;

  ModRefNode()
      : loadFromExternal(NotFlagged),
        storeToExternal(NotFlagged),
        callsExternal(0),
        blocklist(nullptr)
  {}

  // Sets containing PointsToGraph nodes that are loaded from or stored to in this ModRefNode.
  util::HashSet<PointsToGraph::NodeIndex> explicitLoads;
  util::HashSet<PointsToGraph::NodeIndex> explicitStores;

  // If equal to \ref NotFlagged, the flag is considered unset
  // Otherwise, all external memory nodes with size >= the given byte amount are loaded from
  uint32_t loadFromExternal : 15;

  // If equal to \ref NotFlagged, the flag is considered unset
  // Otherwise, all external memory nodes with size >= the given byte amount are stored to
  uint32_t storeToExternal : 15;

  // If set, the Mod/Ref set includes a call to an external function.
  // Setting this also sets the load from/store to flags above to 0 (all external memory)
  uint32_t callsExternal : 1;

  // Optional blocklist that can be added to prevent explicit PointsToGraph nodes from being added.
  // The blocklist is only used during solving, and does not prevent PtG nodes from being added
  // by the addExplicitLoad and addExplicitStore functions
  const util::HashSet<PointsToGraph::NodeIndex> * blocklist;

  /**
   * Adds or updates the flag indicating that this ModRef set possibly loads from all
   * external memory nodes with at least the given byte size.
   * @param minSize the minimum size of memory location loaded from.
   * @return true if the flag was added or updated
   */
  bool
  markAsLoadingFromExternal(size_t minSize)
  {
    if (minSize < loadFromExternal)
    {
      loadFromExternal = minSize;
      return true;
    }
    return false;
  }

  /**
   * Adds or updates the flag indicating that this ModRef set possibly stores to all
   * external memory nodes with at least the given byte size.
   * @param minSize the minimum size of memory location stored to.
   * @return true if the flag was added or updated
   */
  bool
  markAsStoringToExternal(size_t minSize)
  {
    if (minSize < storeToExternal)
    {
      storeToExternal = minSize;
      return true;
    }
    return false;
  }

  [[nodiscard]] std::optional<size_t>
  isLoadingFromExternal() const
  {
    if (loadFromExternal == NotFlagged)
      return std::nullopt;
    return static_cast<size_t>(loadFromExternal);
  }

  [[nodiscard]] std::optional<size_t>
  isStoringToExternal() const
  {
    if (storeToExternal == NotFlagged)
      return std::nullopt;
    return static_cast<size_t>(storeToExternal);
  }
};

/**
 * A graph containing Mod/Ref nodes and constraints
 */
class RegionAwareModRefSummarizer::ModRefGraph final
{
public:
  explicit ModRefGraph(const PointsToGraph & pointsToGraph)
      : pointsToGraph_(pointsToGraph)
  {}

  ModRefGraph(const ModRefGraph &) = delete;
  ModRefGraph &
  operator=(const ModRefGraph &) = delete;

  [[nodiscard]] size_t
  numModRefNodes() const noexcept
  {
    return modRefNodes_.size();
  }

  /**
   * Creates a new ModRefNode that is not mapped to any node
   * @return the index of the new ModRefNode
   */
  [[nodiscard]] ModRefNodeIndex
  createModRefNode()
  {
    modRefNodes_.emplace_back();
    simpleConstraints_.emplace_back();
    return modRefNodes_.size() - 1;
  }

  [[nodiscard]] bool
  hasModRefForNode(const rvsdg::Node & node) const
  {
    return nodeMap_.find(&node) != nodeMap_.end();
  }

  [[nodiscard]] ModRefNodeIndex
  getModRefForNode(const rvsdg::Node & node) const
  {
    const auto it = nodeMap_.find(&node);
    JLM_ASSERT(it != nodeMap_.end());
    return it->second;
  }

  [[nodiscard]] ModRefNodeIndex
  getOrCreateModRefForNode(const rvsdg::Node & node)
  {
    if (const auto it = nodeMap_.find(&node); it != nodeMap_.end())
      return it->second;

    return nodeMap_[&node] = createModRefNode();
  }

  void
  mapNodeToModRef(const rvsdg::Node & node, ModRefNodeIndex index)
  {
    JLM_ASSERT(!hasModRefForNode(node));
    nodeMap_[&node] = index;
  }

  [[nodiscard]] const std::unordered_map<const rvsdg::Node *, ModRefNodeIndex> &
  getNodeMap() const
  {
    return nodeMap_;
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  getExplicitLoads(ModRefNodeIndex modRefNode) const
  {
    return modRefNodes_[modRefNode].explicitLoads;
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  getExplicitStores(ModRefNodeIndex modRefNode) const
  {
    return modRefNodes_[modRefNode].explicitStores;
  }

  /**
   * Inserts the given \p ptgNode as an explicit load in the given \p modRefNode.
   * Ignores the loadFromExternal flag and the blocklist.
   * @param modRefNode the index of the modRefNode
   * @param ptgNode the index of the PointsToGraph Node
   * @return true if the ptgNode was added as an explicit load, false otherwise.
   */
  bool
  addExplicitLoad(ModRefNodeIndex modRefNode, PointsToGraph::NodeIndex ptgNode)
  {
    JLM_ASSERT(modRefNode < modRefNodes_.size());
    return modRefNodes_[modRefNode].explicitLoads.insert(ptgNode);
  }

  /**
   * Inserts the given \p ptgNode as an explicit store in the given \p modRefNode.
   * Ignores the storeToExternal flag and the blocklist.
   * @param modRefNode the index of the modRefNode
   * @param ptgNode the index of the PointsToGraph Node
   * @return true if the ptgNode was added as an explicit store, false otherwise.
   */
  bool
  addExplicitStore(ModRefNodeIndex modRefNode, PointsToGraph::NodeIndex ptgNode)
  {
    JLM_ASSERT(modRefNode < modRefNodes_.size());
    return modRefNodes_[modRefNode].explicitStores.insert(ptgNode);
  }

  [[nodiscard]] std::optional<size_t>
  isLoadingFromExternal(ModRefNodeIndex modRefNode) const
  {
    JLM_ASSERT(modRefNode < modRefNodes_.size());
    return modRefNodes_[modRefNode].isLoadingFromExternal();
  }

  [[nodiscard]] std::optional<size_t>
  isStoringToExternal(ModRefNodeIndex modRefNode) const
  {
    JLM_ASSERT(modRefNode < modRefNodes_.size());
    return modRefNodes_[modRefNode].isStoringToExternal();
  }

  /**
   * Marks the given \p modRefNode as loading from all external memory [of at least the given size]
   * @param modRefNode the ModRefNode that gains the flag.
   * @param minSize [optional] the minimum byte size of the external memory being loaded from.
   * @return true if the modRefNode gained a (stricter) flag, false otherwise.
   */
  bool
  markAsLoadingFromExternal(ModRefNodeIndex modRefNode, std::optional<size_t> minSize)
  {
    JLM_ASSERT(modRefNode < modRefNodes_.size());
    return modRefNodes_[modRefNode].markAsLoadingFromExternal(minSize.value_or(0));
  }

  /**
   * Marks the given \p modRefNode as storing to all external memory [with at least the given size]
   * @param modRefNode the ModRefNode that gains the flag.
   * @param minSize [optional] the minimum byte size of the external memory being stored to.
   * @return true if the modRefNode gained a (stricter) flag, false otherwise.
   */
  bool
  markAsStoringToExternal(ModRefNodeIndex modRefNode, std::optional<size_t> minSize)
  {
    JLM_ASSERT(modRefNode < modRefNodes_.size());
    return modRefNodes_[modRefNode].markAsStoringToExternal(minSize.value_or(0));
  }

  [[nodiscard]] bool
  isCallingExternal(ModRefNodeIndex modRefNode) const
  {
    JLM_ASSERT(modRefNode < modRefNodes_.size());
    return modRefNodes_[modRefNode].callsExternal;
  }

  bool
  markAsCallingExternal(ModRefNodeIndex modRefNode)
  {
    JLM_ASSERT(modRefNode < modRefNodes_.size());
    if (modRefNodes_[modRefNode].callsExternal)
      return false;
    modRefNodes_[modRefNode].callsExternal = 1;
    modRefNodes_[modRefNode].loadFromExternal = 0;
    modRefNodes_[modRefNode].storeToExternal = 0;
    return true;
  }

  /**
   * Adds the simple constraint edge from -> to, which causes "to" to be a superset of "from"
   * in the final solution. The solution will respect a blocklist in the "to" node, however.
   * @param from the index of the ModRefNode to propagate from
   * @param to the index of the ModRefNode to propagate to
   */
  void
  addSimpleConstraintEdge(ModRefNodeIndex from, ModRefNodeIndex to)
  {
    simpleConstraints_[from].push_back(to);
  }

  /**
   * Defines a blocklist for the ModRefNode with the given \p index.
   * The set can not have any explicit loads or stores, and can not already have a blocklist.
   * @param index the ModRefNode that gains a blocklist
   * @param blocklist the list of PointsToGraph nodes to block.
   * The reference must stay valid until after solving has finished.
   */
  void
  addBlocklist(ModRefNodeIndex index, const util::HashSet<PointsToGraph::NodeIndex> & blocklist)
  {
    JLM_ASSERT(index < modRefNodes_.size());
    if (modRefNodes_[index].blocklist != nullptr)
      throw std::logic_error("Blocklist already set");
    if (!modRefNodes_[index].explicitLoads.IsEmpty())
      throw std::logic_error("Blocklist cannot be set on ModRefNode with explicit loads");
    if (!modRefNodes_[index].explicitStores.IsEmpty())
      throw std::logic_error("Blocklist cannot be set on ModRefNode with explicit stores");

    modRefNodes_[index].blocklist = &blocklist;
  }

  /**
   * Solves the ModRefGraph by propagating between ModRefNodes until all constraints are satisfied,
   * while also respecting blocklists.
   */
  void
  solveConstraints()
  {
    util::FifoWorklist<ModRefNodeIndex> worklist;

    // Start by pushing everything to the worklist
    for (ModRefNodeIndex i = 0; i < numModRefNodes(); i++)
      worklist.PushWorkItem(i);

    while (worklist.HasMoreWorkItems())
    {
      const auto workItem = worklist.PopWorkItem();

      // Handle all simple constraints workItem -> target
      for (auto target : simpleConstraints_[workItem])
      {
        if (propagateModRefNode(workItem, target))
          worklist.PushWorkItem(target);
      }
    }

    JLM_ASSERT(verifyBlocklists());
  }

  /**
   * Produce a textual representation of the ModRefNode with the given \p index.
   * Includes flags and explicit loads and stores.
   * @param index the index of the ModRefNode to produce the debug string for
   * @return
   */
  [[nodiscard]] std::string
  debugStringForSet(ModRefNodeIndex index) const
  {
    JLM_ASSERT(index < modRefNodes_.size());

    std::ostringstream ss;
    ss << "(MRNode#" << std::setfill('0') << std::setw(3) << index << std::setw(0);
    ss << " loads:{";
    bool printSep = false;
    for (const auto load : modRefNodes_[index].explicitLoads.Items())
    {
      if (printSep)
        ss << ", ";
      else
        printSep = true;
      ss << pointsToGraph_.getNodeDebugString(load);
    }
    ss << "} stores:{";
    printSep = false;
    for (const auto store : modRefNodes_[index].explicitStores.Items())
    {
      if (printSep)
        ss << ", ";
      else
        printSep = true;
      ss << pointsToGraph_.getNodeDebugString(store);
    }
    ss << "}";

    const auto loadingFromExternal = modRefNodes_[index].isLoadingFromExternal();
    const auto storingToExternal = modRefNodes_[index].isStoringToExternal();
    if (loadingFromExternal)
    {
      ss << " (lFromExt>=" << *loadingFromExternal << ")";
    }
    if (storingToExternal)
    {
      ss << " (sToExtr>= " << *storingToExternal << ")";
    }
    if (modRefNodes_[index].callsExternal)
    {
      ss << " (callsExt)";
    }
    ss << " )";

    return ss.str();
  }

private:
  /**
   * Propagates flags and explicit loads and stores from one ModRefNode to another.
   * Respects the blocklist of the \p to set, if it has one.
   * @param from the ModRefNode to propagate from
   * @param to the ModRefNode to propagate to
   * @return true if the \p to ModRefNode gained any flags or explicit targets, false otherwise
   */
  bool
  propagateModRefNode(ModRefNodeIndex from, ModRefNodeIndex to)
  {
    JLM_ASSERT(from < modRefNodes_.size());
    JLM_ASSERT(to < modRefNodes_.size());

    bool changed = false;

    if (!modRefNodes_[to].callsExternal && modRefNodes_[from].callsExternal)
    {
      modRefNodes_[to].callsExternal = true;
      changed = true;
    }
    changed |= modRefNodes_[to].markAsLoadingFromExternal(modRefNodes_[from].loadFromExternal);
    changed |= modRefNodes_[to].markAsStoringToExternal(modRefNodes_[from].storeToExternal);

    const auto blocklist = modRefNodes_[to].blocklist;
    const auto loadFromExternal = modRefNodes_[to].isLoadingFromExternal();
    const auto storeToExternal = modRefNodes_[to].isStoringToExternal();

    // Propagate explicit loads and stores
    for (auto load : modRefNodes_[from].explicitLoads.Items())
    {
      if (blocklist && blocklist->Contains(load))
        continue;

      if (ENABLE_MOD_REF_PIP && loadFromExternal && pointsToGraph_.isExternallyAvailable(load))
      {
        // The "to" set is already flagged as loading from all external memory above a given size
        // check if "load" is already included.
        const auto loadSize = pointsToGraph_.tryGetNodeSize(load);
        if (!loadSize || *loadSize >= *loadFromExternal)
          continue;
      }

      changed |= addExplicitLoad(to, load);
    }
    for (auto store : modRefNodes_[from].explicitStores.Items())
    {
      if (blocklist && blocklist->Contains(store))
        continue;

      if (ENABLE_MOD_REF_PIP && storeToExternal && pointsToGraph_.isExternallyAvailable(store))
      {
        // The "to" set is already flagged as storing from all external memory above a given size
        // check if "store" is already included.
        const auto storeSize = pointsToGraph_.tryGetNodeSize(store);
        if (!storeSize || *storeSize >= *storeToExternal)
          continue;
      }

      changed |= addExplicitStore(to, store);
    }

    return changed;
  }

  /**
   * Internal sanity check. After solving, ModRefNodes with blocklists should not
   * contain any of their blocked PointsToGraph nodes in their sets of explicit loads or stores.
   * @return true if validation passed for all ModRefNodes with blocklists, otherwise false.
   */
  [[nodiscard]] bool
  verifyBlocklists() const
  {
    for (ModRefNodeIndex index = 0; index < numModRefNodes(); index++)
    {
      const auto blocklist = modRefNodes_[index].blocklist;
      if (blocklist == nullptr)
        continue;

      for (auto ptgNode : modRefNodes_[index].explicitLoads.Items())
      {
        if (blocklist->Contains(ptgNode))
          return false;
      }
      for (auto ptgNode : modRefNodes_[index].explicitStores.Items())
      {
        if (blocklist->Contains(ptgNode))
          return false;
      }
    }
    return true;
  }

  /**
   * A reference to the PointsToGraph used to create the Mod/Ref sets.
   */
  const PointsToGraph & pointsToGraph_;

  /**
   * All ModRefNodes in the constraint graph
   */
  std::vector<ModRefNode> modRefNodes_;

  /**
   * Map from RVSDG Nodes that have memory side effects, to their ModRefNode.
   * Includes nodes like loads, stores, memcpy, free and calls.
   * Also includes structural nodes like gamma, theta and lambda.
   */
  std::unordered_map<const rvsdg::Node *, ModRefNodeIndex> nodeMap_;

  /**
   * Simple edges in the constraint graph.
   * A simple edge a -> b indicates that the ModRefNode b should be a superset of set a.
   * If b has a blocklist, it is respected.
   * This list is always the same length as modRefNodes_.
   */
  std::vector<std::vector<ModRefNodeIndex>> simpleConstraints_;
};

/**
 * Contains metadata about the MemoryNodeOrdering
 */
struct RegionAwareModRefSummarizer::MemoryNodeOrderingMetadata
{
  // For each memory node size that appears among externally available memory nodes,
  // this map provides the lowest index for each memory node size.
  // Memory nodes with unknown size are given the maximum possible size_t.
  std::map<size_t, MemoryNodeOrderingIndex> firstExternallyAvailableWithSize;

  // The first memory node that is not externally available
  MemoryNodeOrderingIndex endOfExternallyAvailable = 0;

  //
  MemoryNodeOrderingIndex startOfStoredByExternal = 0;
  MemoryNodeOrderingIndex endOfStoredByExternal = 0;

  MemoryNodeOrderingIndex startOfLoadedByExternal = 0;
  MemoryNodeOrderingIndex endOfLoadedByExternal = 0;

  std::vector<MemoryNodeOrderingIndex> ptgNodeIndexToMemoryOrderingIndex;
};

/**
 * Struct holding temporary data used during the creation of a single mod/ref summary
 */
struct RegionAwareModRefSummarizer::Context
{
  explicit Context(const PointsToGraph & pointsToGraph)
      : pointsToGraph(pointsToGraph),
        modRefGraph(pointsToGraph)
  {}

  /**
   * The PointsToGraph used to create the Mod/Ref summaries
   */
  const PointsToGraph & pointsToGraph;

  /**
   * The set of functions belonging to each SCC in the call graph.
   * The SCCs are ordered in reverse topological order, so
   * if function a() calls b(), and they are not in the same SCC,
   * the SCC containing a() comes after the SCC containing b().
   *
   * External functions are not included in these sets, see \ref ExternalNodeSccIndex.
   *
   * Assigned in \ref CreateCallGraph(). Remains constant after.
   */
  std::vector<util::HashSet<const rvsdg::LambdaNode *>> SccFunctions;

  /**
   * The index of the SCC in the call graph that represent containing all external functions
   *
   * Assigned in \ref CreateCallGraph(). Remains constant after.
   */
  size_t ExternalNodeSccIndex = 0;

  /**
   * For each SCC in the call graph, the set of SCCs it targets using calls.
   * Since SCCs are ordered in reverse topological order, an SCC never targets higher indices.
   * If there is any possibility of recursion within an SCC, it also targets itself.
   *
   * Assigned in \ref CreateCallGraph(). Remains constant after.
   */
  std::vector<util::HashSet<size_t>> SccCallTargets;

  /**
   * A mapping from functions to the index of the SCC they belong to in the call graph
   *
   * Assigned in \ref CreateCallGraph(). Remains constant after.
   */
  std::unordered_map<const rvsdg::LambdaNode *, size_t> FunctionToSccIndex;

  /**
   * For each SCC, only allocas defined within the SCC, or within a predecessor of the SCC,
   * can possibly be live. All other allocas are considered dead in the SCC.
   *
   * Assigned in \ref FindAllocasDeadInSccs(). Remains constant after.
   */
  std::vector<util::HashSet<PointsToGraph::NodeIndex>> AllocasDeadInScc;

  /**
   * The set of all Simple Allocas in the module.
   *
   * Assigned in \ref CreateSimpleAllocaSet(). Remains constant after.
   */
  util::HashSet<PointsToGraph::NodeIndex> simpleAllocas;

  /**
   * For each region, this field contains the set of allocas defined in the region,
   * that have been show to be Non-Reentrant.
   *
   * Assigned in \ref CreateNonReentrantAllocaSets(). Remains constant after.
   */
  std::unordered_map<const rvsdg::Region *, util::HashSet<PointsToGraph::NodeIndex>>
      NonReentrantAllocas;

  /**
   * Graph containing ModRef sets and their constraints.
   */
  ModRefGraph modRefGraph;

  /**
   * A ModRefNode containing memory nodes that can be written to or read from by external functions.
   * Externally available memory is automatically included, but we must also account for memory
   * that never escapes the module, but can be changed in functions called from external code.
   * All such memory nodes end up as explicit loads and stores in this ModRefNode.
   *
   * Assigned in \ref createExternalModRefNode(). Remains constant after.
   */
  ModRefNodeIndex externalModRefNode = 0;

  /**
   * The ordering of all relevant memory nodes, to be used by the final ModRefSummary
   */
  std::unique_ptr<MemoryNodeOrdering> memoryNodeOrdering;

  /**
   * Metadata about the memoryNodeOrdering, used to convert ModRefNodes to ModRefSets
   */
  MemoryNodeOrderingMetadata memoryNodeOrderingMetadata;
};

/**
 * Class holding the final result of the whole RegionAwareModRefSummarizer process.
 */
class RegionAwareModRefSummarizer::RegionAwareModRefSummary final : public ModRefSummary
{
public:
  using ModRefSetIndex = ModRefNodeIndex;

  RegionAwareModRefSummary(
      MemoryNodeOrdering memoryNodeOrdering,
      std::vector<ModRefSet> modRefSets,
      std::unordered_map<const rvsdg::Node *, ModRefSetIndex> nodeMap)
      : memoryNodeOrdering_(std::move(memoryNodeOrdering)),
        modRefSets_(std::move(modRefSets)),
        nodeMap_(std::move(nodeMap))
  {}

  const MemoryNodeOrdering &
  getMemoryNodeOrdering() const override
  {
    return memoryNodeOrdering_;
  }

  const ModRefSet &
  getSimpleNodeModRef(const rvsdg::SimpleNode & node) const override
  {
    if (const auto it = nodeMap_.find(&node); it != nodeMap_.end())
      return modRefSets_[it->second];

    throw std::logic_error("Node not found in ModRefSummary");
  }

  const ModRefSet &
  getGammaEntryModRef(const rvsdg::GammaNode & gamma) const override
  {
    if (const auto it = nodeMap_.find(&gamma); it != nodeMap_.end())
      return modRefSets_[it->second];

    throw std::logic_error("GammaNode not found in ModRefSummary");
  }

  const ModRefSet &
  getGammaExitModRef(const rvsdg::GammaNode & gamma) const override
  {
    if (const auto it = nodeMap_.find(&gamma); it != nodeMap_.end())
      return modRefSets_[it->second];

    throw std::logic_error("GammaNode not found in ModRefSummary");
  }

  const ModRefSet &
  getThetaModRef(const rvsdg::ThetaNode & theta) const override
  {
    if (const auto it = nodeMap_.find(&theta); it != nodeMap_.end())
      return modRefSets_[it->second];

    throw std::logic_error("ThetaNode not found in ModRefSummary");
  }

  const ModRefSet &
  getLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const override
  {
    if (const auto it = nodeMap_.find(&lambda); it != nodeMap_.end())
      return modRefSets_[it->second];

    throw std::logic_error("LambdaNode not found in ModRefSummary");
  }

  const ModRefSet &
  getLambdaExitModRef(const rvsdg::LambdaNode & lambda) const override
  {
    if (const auto it = nodeMap_.find(&lambda); it != nodeMap_.end())
      return modRefSets_[it->second];

    throw std::logic_error("LambdaNode not found in ModRefSummary");
  }

private:
  // The memory node ordering used by all ModRefSets
  MemoryNodeOrdering memoryNodeOrdering_;

  // The ModRef sets in the
  std::vector<ModRefSet> modRefSets_;

  std::unordered_map<const rvsdg::Node *, ModRefSetIndex> nodeMap_;
};

RegionAwareModRefSummarizer::~RegionAwareModRefSummarizer() noexcept = default;

RegionAwareModRefSummarizer::RegionAwareModRefSummarizer() = default;

std::unique_ptr<ModRefSummary>
RegionAwareModRefSummarizer::SummarizeModRefs(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{

  Context_ = std::make_unique<Context>(pointsToGraph);
  auto statistics = Statistics::Create(rvsdgModule, pointsToGraph);

  statistics->StartCallGraphStatistics();
  createCallGraph(rvsdgModule);
  statistics->StopCallGraphStatistics(Context_->SccFunctions.size());

  statistics->StartAllocasDeadInSccStatistics();
  findAllocasDeadInSccs();
  statistics->StopAllocasDeadInSccStatistics();

  statistics->StartCreateSimpleAllocasSetStatistics();
  Context_->simpleAllocas = createSimpleAllocaSet(pointsToGraph);
  statistics->StopCreateSimpleAllocasSetStatistics(Context_->simpleAllocas.Size());

  statistics->StartCreateNonReentrantAllocaSetsStatistics();
  auto numNonReentrantAllocas = createNonReentrantAllocaSets();
  statistics->StopCreateNonReentrantAllocaSetsStatistics(numNonReentrantAllocas);

  statistics->StartAnnotationStatistics();
  // Go through and recursively annotate all functions, regions and nodes
  for (const auto & scc : Context_->SccFunctions)
  {
    for (const auto lambda : scc.Items())
    {
      annotateFunction(*lambda);
    }
  }
  statistics->StopAnnotationStatistics();

  statistics->StartCreateExternalModRefNode();
  createExternalModRefNode();
  statistics->StopCreateExternalModRefNode();

  statistics->StartSolvingStatistics();
  Context_->modRefGraph.solveConstraints();
  statistics->StopSolvingStatistics();

  // Print debug output
  std::cerr << PointsToGraph::ToDot(pointsToGraph) << std::endl;
  std::cerr << "numSimpleAllocas: " << Context_->simpleAllocas.Size() << std::endl;
  std::cerr << "numNonReentrantAllocas: " << numNonReentrantAllocas << std::endl;
  std::cerr << "Call Graph SCCs:" << std::endl << callGraphSCCsToString() << std::endl;
  std::cerr << "RegionTree:" << std::endl << dumpRegionTree(rvsdgModule.Rvsdg()) << std::endl;

  statistics->StartCreateMemoryNodeOrderingStatistics();
  createMemoryNodeOrdering();
  statistics->StopCreateMemoryNodeOrderingStatistics();

  statistics->StartCreateModRefSummaryStatistics();
  auto result = createModRefSummary();
  statistics->StopCreateModRefSummaryStatistics();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
  Context_.reset();
  return result;
}

/**
 * Collects all lambda nodes defined in the given module, in an unspecified order.
 * @param rvsdgModule the module
 * @return a list of all lambda nodes in the module
 */
static std::vector<const rvsdg::LambdaNode *>
collectLambdaNodes(const rvsdg::RvsdgModule & rvsdgModule)
{
  std::vector<const rvsdg::LambdaNode *> result;

  // Recursively traverses all structural nodes, but does not enter into lambdas
  const std::function<void(rvsdg::Region &)> collectLambdasInRegion =
      [&](rvsdg::Region & region) -> void
  {
    for (auto & node : region.Nodes())
    {
      if (auto lambda = dynamic_cast<rvsdg::LambdaNode *>(&node))
      {
        result.push_back(lambda);
      }
      else if (auto structural = dynamic_cast<rvsdg::StructuralNode *>(&node))
      {
        for (size_t i = 0; i < structural->nsubregions(); i++)
        {
          collectLambdasInRegion(*structural->subregion(i));
        }
      }
    }
  };

  collectLambdasInRegion(rvsdgModule.Rvsdg().GetRootRegion());

  return result;
}

void
RegionAwareModRefSummarizer::createCallGraph(const rvsdg::RvsdgModule & rvsdgModule)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // The list of lambdas becomes the list of nodes in the call graph
  auto lambdaNodes = collectLambdaNodes(rvsdgModule);

  // Mapping from LambdaNode* to its index in lambdaNodes
  std::unordered_map<const rvsdg::LambdaNode *, size_t> callGraphNodeIndex;
  callGraphNodeIndex.reserve(lambdaNodes.size());
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    callGraphNodeIndex.insert({ lambdaNodes[i], i });
  }

  // Add a dummy node representing all external functions, with no associated LambdaNode
  const auto externalNodeIndex = lambdaNodes.size();
  const auto numCallGraphNodes = externalNodeIndex + 1;

  // Outgoing edges for each node in the call graph
  std::vector<util::HashSet<size_t>> callGraphSuccessors(numCallGraphNodes);

  // Add outgoing edges from the given caller to any function the call may target
  const auto HandleCall = [&](rvsdg::Node & callNode, size_t callerIndex) -> void
  {
    JLM_ASSERT(is<CallOperation>(&callNode));
    const auto targetPtr = callNode.input(0)->origin();
    const auto targetPtrPtgNode = pointsToGraph.getRegisterNode(*targetPtr);

    // Go through all locations the called function pointer may target
    for (auto calleePtgNode : pointsToGraph.getTargets(targetPtrPtgNode).Items())
    {
      auto kind = pointsToGraph.getKind(calleePtgNode);
      if (kind == PointsToGraph::NodeKind::LambdaNode)
      {
        const auto & lambdaNode = pointsToGraph.getLambdaNodeObject(calleePtgNode);

        // Look up which call graph node represents the target lambda
        JLM_ASSERT(callGraphNodeIndex.find(&lambdaNode) != callGraphNodeIndex.end());
        const auto calleeCallGraphNode = callGraphNodeIndex[&lambdaNode];

        // Add the edge caller -> callee to the call graph
        callGraphSuccessors[callerIndex].insert(calleeCallGraphNode);
      }
      else if (kind == PointsToGraph::NodeKind::ImportNode)
      {
        // Add the edge caller -> node representing external functions
        callGraphSuccessors[callerIndex].insert(externalNodeIndex);
      }
    }
    if (pointsToGraph.isTargetingAllExternallyAvailable(targetPtrPtgNode))
    {
      // If the call target pointer is flagged, add an edge to external functions
      callGraphSuccessors[callerIndex].insert(externalNodeIndex);
    }
  };

  // Recursive function finding all call operations, adding edges to the call graph
  const std::function<void(rvsdg::Region &, size_t)> handleCalls = [&](rvsdg::Region & region,
                                                                       size_t callerIndex) -> void
  {
    for (auto & node : region.Nodes())
    {
      if (is<CallOperation>(&node))
      {
        HandleCall(node, callerIndex);
      }
      else if (auto structural = dynamic_cast<rvsdg::StructuralNode *>(&node))
      {
        for (size_t i = 0; i < structural->nsubregions(); i++)
        {
          handleCalls(*structural->subregion(i), callerIndex);
        }
      }
    }
  };

  // For all functions, visit all their calls and add outgoing edges in the call graph
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    handleCalls(*lambdaNodes[i]->subregion(), i);

    // If the function has escaped, add an edge from the node representing all external functions
    const auto lambdaPtgNode = pointsToGraph.getLambdaNode(*lambdaNodes[i]);
    if (pointsToGraph.isExternallyAvailable(lambdaPtgNode))
    {
      callGraphSuccessors[externalNodeIndex].insert(i);
    }
  }

  // Finally add the fact that the external node may call itself
  callGraphSuccessors[externalNodeIndex].insert(externalNodeIndex);

  // Used by the implementation of Tarjan's SCC algorithm
  const auto GetSuccessors = [&](size_t nodeIndex)
  {
    return callGraphSuccessors[nodeIndex].Items();
  };

  // Find SCCs in the call graph
  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSCCs = util::FindStronglyConnectedComponents<size_t>(
      numCallGraphNodes,
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);

  // sccIndex are distributed in a reverse topological order, so the sccIndex is used
  // when creating the list of SCCs and the functions they contain
  Context_->SccFunctions.resize(numSCCs);
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    Context_->SccFunctions[sccIndex[i]].insert(lambdaNodes[i]);
    Context_->FunctionToSccIndex[lambdaNodes[i]] = sccIndex[i];
  }

  // Add edges between the SCCs for all calls
  Context_->SccCallTargets.resize(numSCCs);
  for (size_t i = 0; i < numCallGraphNodes; i++)
  {
    for (auto target : callGraphSuccessors[i].Items())
    {
      Context_->SccCallTargets[sccIndex[i]].insert(sccIndex[target]);
    }
  }

  // Also note which SCC contains all external functions
  Context_->ExternalNodeSccIndex = sccIndex[externalNodeIndex];
}

[[nodiscard]] static const rvsdg::LambdaNode &
getSurroundingLambdaNode(const rvsdg::Node & node)
{
  auto it = &node;
  while (it)
  {
    if (auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(it))
      return *lambda;
    it = it->region()->node();
  }
  JLM_UNREACHABLE("node was not in a lambda");
}

void
RegionAwareModRefSummarizer::findAllocasDeadInSccs()
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // First find which allocas may be live in each SCC
  std::vector<util::HashSet<PointsToGraph::NodeIndex>> liveAllocas(Context_->SccFunctions.size());

  util::HashSet<PointsToGraph::NodeIndex> allAllocas;

  // Add all Allocas to the SCC of the function they are defined in
  for (auto allocaPtgNode : pointsToGraph.allocaNodes())
  {
    allAllocas.insert(allocaPtgNode);
    const auto & allocaNode = pointsToGraph.getAllocaNodeObject(allocaPtgNode);
    const auto & lambdaNode = getSurroundingLambdaNode(allocaNode);
    JLM_ASSERT(Context_->FunctionToSccIndex.count(&lambdaNode));
    const auto sccIndex = Context_->FunctionToSccIndex[&lambdaNode];
    liveAllocas[sccIndex].insert(allocaPtgNode);
  }

  // Propagate live allocas to targets of function calls.
  // I.e., if a() -> b(), then any alloca that is live in a() may also be live in b()
  // Start at the topologically earliest SCC, which has the largest SCC index
  // The topologically latest SCC can't have any targets
  for (size_t sccIndex = Context_->SccFunctions.size() - 1; sccIndex > 0; sccIndex--)
  {
    for (auto targetScc : Context_->SccCallTargets[sccIndex].Items())
    {
      JLM_ASSERT(targetScc <= sccIndex);
      if (targetScc != sccIndex)
        liveAllocas[targetScc].UnionWith(liveAllocas[sccIndex]);
    }
  }

  // Any alloca that is not live in the SCC gets added to the set of allocas that are dead
  Context_->AllocasDeadInScc.resize(Context_->SccFunctions.size());
  for (size_t sccIndex = 0; sccIndex < Context_->SccFunctions.size(); sccIndex++)
  {
    Context_->AllocasDeadInScc[sccIndex].UnionWith(allAllocas);
    Context_->AllocasDeadInScc[sccIndex].DifferenceWith(liveAllocas[sccIndex]);
  }
}

util::HashSet<PointsToGraph::NodeIndex>
RegionAwareModRefSummarizer::createSimpleAllocaSet(const PointsToGraph & pointsToGraph)
{
  // The set of allocas that are simple. Starts off as an over-approximation
  util::HashSet<PointsToGraph::NodeIndex> simpleAllocas;
  // A queue used to visit all PtG memory nodes that are not simple allocas
  std::queue<PointsToGraph::NodeIndex> notSimple;

  for (PointsToGraph::NodeIndex ptgNode = 0; ptgNode < pointsToGraph.numNodes(); ptgNode++)
  {
    // Ignore register nodes. Only memory nodes are relevant
    if (pointsToGraph.isRegisterNode(ptgNode))
      continue;

    if (pointsToGraph.getKind(ptgNode) == PointsToGraph::NodeKind::AllocaNode
        && !pointsToGraph.isExternallyAvailable(ptgNode))
      simpleAllocas.insert(ptgNode);
    else
      notSimple.push(ptgNode);
  }

  // Process the queue to visit all memory nodes that may disqualify allocas from being simple
  while (!notSimple.empty())
  {
    const auto ptgNode = notSimple.front();
    notSimple.pop();

    // Any node targeted by the not-simple memory node can themselves not be simple
    for (const auto targetPtgNode : pointsToGraph.getTargets(ptgNode).Items())
    {
      // If the target is currently in the simple allocas candiate set, move it to the queue
      if (simpleAllocas.Remove(targetPtgNode))
        notSimple.push(targetPtgNode);
    }
  }

  return simpleAllocas;
}

/**
 * Gets the set of simple alloca PtG nodes that it is possible to reach from region arguments.
 * Reachability is defined in terms of the PointsToGraph. A simple alloca is by definition
 * only reachable from RegisterNodes and other simple allocas,
 * so other types of PointsToGraph node can be ignored.
 * @param region the region whose arguments are checked
 * @return the set of simple allocas reachable from region arguments
 */
util::HashSet<PointsToGraph::NodeIndex>
RegionAwareModRefSummarizer::getSimpleAllocasReachableFromRegionArguments(
    const rvsdg::Region & region)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // Use a queue and a set to traverse the PointsToGraph
  util::HashSet<PointsToGraph::NodeIndex> reachableSimpleAllocas;
  std::queue<PointsToGraph::NodeIndex> nodes;
  for (auto argument : region.Arguments())
  {
    if (!IsPointerCompatible(*argument))
      continue;
    const auto ptgNode = pointsToGraph.getRegisterNode(*argument);
    nodes.push(ptgNode);
  }

  // Traverse along PointsToGraph edges to find all reachable simple allocas
  while (!nodes.empty())
  {
    auto ptgNode = nodes.front();
    nodes.pop();

    for (auto & targetPtgNode : pointsToGraph.getTargets(ptgNode).Items())
    {
      // We only are about following simple allocas, as simple allocas are only reachable from them.
      if (!Context_->simpleAllocas.Contains(targetPtgNode))
        continue;

      if (reachableSimpleAllocas.insert(targetPtgNode))
        nodes.push(targetPtgNode);
    }
  }

  return reachableSimpleAllocas;
}

bool
RegionAwareModRefSummarizer::isRecursionPossible(const rvsdg::LambdaNode & lambda)
{
  const auto scc = Context_->FunctionToSccIndex[&lambda];
  return Context_->SccCallTargets[scc].Contains(scc);
}

size_t
RegionAwareModRefSummarizer::createNonReentrantAllocaSets()
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  size_t numNonReentrantAllocas = 0;

  std::unordered_map<const rvsdg::Region *, util::HashSet<PointsToGraph::NodeIndex>>
      reachableSimpleAllocas;

  const auto getReachableSimpleAllocas =
      [&](const rvsdg::Region & region) -> const util::HashSet<PointsToGraph::NodeIndex> &
  {
    if (const auto it = reachableSimpleAllocas.find(&region); it != reachableSimpleAllocas.end())
    {
      return it->second;
    }
    return reachableSimpleAllocas[&region] = getSimpleAllocasReachableFromRegionArguments(region);
  };

  // Only simple allocas are candidates for being Non-Reentrant
  for (auto simpleAllocaPtgNode : Context_->simpleAllocas.Items())
  {
    auto & allocaNode = pointsToGraph.getAllocaNodeObject(simpleAllocaPtgNode);
    const auto & region = *allocaNode.region();

    // If the alloca's function is never involved in any recursion,
    // the alloca is definitely non-reentrant.
    const auto & lambda = getSurroundingLambdaNode(allocaNode);
    if (isRecursionPossible(lambda))
    {
      // In lambdas where recursion is possible, only simple allocas that are provably not
      // passed in through region arguments can be considered Non-Reentrant
      const auto & reachable = getReachableSimpleAllocas(region);
      if (reachable.Contains(simpleAllocaPtgNode))
        continue;
    }

    // Creates a set for the region if it does not already have one, and add the alloca
    Context_->NonReentrantAllocas[&region].insert(simpleAllocaPtgNode);
    numNonReentrantAllocas++;
  }

  return numNonReentrantAllocas;
}

void
RegionAwareModRefSummarizer::annotateFunction(const rvsdg::LambdaNode & lambda)
{
  const auto & region = *lambda.subregion();
  const auto regionModRefNode = annotateRegion(region, lambda);

  // Due to calls targeting this function, the lambda may already have a Mod/Ref set
  // If so, create an edge between the region Mod/Ref set and the lambda Mod/Ref set
  // otherwise, just map the lambda to the region's Mod/Ref set
  if (Context_->modRefGraph.hasModRefForNode(lambda))
  {
    const auto lambdaModRefNode = Context_->modRefGraph.getOrCreateModRefForNode(lambda);
    Context_->modRefGraph.addSimpleConstraintEdge(regionModRefNode, lambdaModRefNode);
  }
  else
  {
    Context_->modRefGraph.mapNodeToModRef(lambda, regionModRefNode);
  }
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateRegion(
    const rvsdg::Region & region,
    const rvsdg::LambdaNode & lambda)
{
  const auto regionModRefNode = Context_->modRefGraph.createModRefNode();

  for (auto & node : region.Nodes())
  {
    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      const auto nodeModRefNode = annotateStructuralNode(*structuralNode, lambda);
      Context_->modRefGraph.addSimpleConstraintEdge(nodeModRefNode, regionModRefNode);
    }
    else if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
    {
      const auto nodeModRefNode = annotateSimpleNode(*simpleNode, lambda);
      if (nodeModRefNode)
        Context_->modRefGraph.addSimpleConstraintEdge(*nodeModRefNode, regionModRefNode);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }

  // Check if this region has any Non-Reentrant Allocas. If so, block them
  if (const auto it = Context_->NonReentrantAllocas.find(&region);
      it != Context_->NonReentrantAllocas.end() && ENABLE_NON_REENTRANT_ALLOCA_BLOCKLIST)
  {
    Context_->modRefGraph.addBlocklist(regionModRefNode, it->second);
  }

  return regionModRefNode;
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateStructuralNode(
    const rvsdg::StructuralNode & structuralNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRefNode = Context_->modRefGraph.getOrCreateModRefForNode(structuralNode);

  for (auto & subregion : structuralNode.Subregions())
  {
    const auto subregionModRefSef = annotateRegion(subregion, lambda);
    Context_->modRefGraph.addSimpleConstraintEdge(subregionModRefSef, nodeModRefNode);
  }

  return nodeModRefNode;
}

std::optional<ModRefNodeIndex>
RegionAwareModRefSummarizer::annotateSimpleNode(
    const rvsdg::SimpleNode & simpleNode,
    const rvsdg::LambdaNode & lambda)
{
  if (is<LoadOperation>(&simpleNode))
    return annotateLoad(simpleNode, lambda);

  if (is<StoreOperation>(&simpleNode))
    return annotateStore(simpleNode, lambda);

  if (is<AllocaOperation>(&simpleNode))
    return annotateAlloca(simpleNode);

  if (is<MallocOperation>(&simpleNode))
    return annotateMalloc(simpleNode);

  if (is<FreeOperation>(&simpleNode))
    return annotateFree(simpleNode, lambda);

  if (is<MemCpyOperation>(&simpleNode))
    return annotateMemcpy(simpleNode, lambda);

  if (is<CallOperation>(&simpleNode))
    return annotateCall(simpleNode, lambda);

  return std::nullopt;
}

template<bool IsStore>
void
RegionAwareModRefSummarizer::addPointerOriginTargets(
    ModRefNodeIndex modRefNode,
    const rvsdg::Output & origin,
    std::optional<size_t> minTargetSize,
    const rvsdg::LambdaNode & lambda)
{
  const auto & pointsToGraph = Context_->pointsToGraph;
  auto & modRefGraph = Context_->modRefGraph;

  // TODO: Re-use ModRefNodes for all uses of the registerNode in this function
  const auto & registerPtgNode = pointsToGraph.getRegisterNode(origin);

  if (pointsToGraph.isTargetingAllExternallyAvailable(registerPtgNode))
  {
    if constexpr (IsStore)
      modRefGraph.markAsStoringToExternal(modRefNode, minTargetSize);
    else
      modRefGraph.markAsLoadingFromExternal(modRefNode, minTargetSize);
  }

  const auto & allocasDead = Context_->AllocasDeadInScc[Context_->FunctionToSccIndex[&lambda]];
  for (const auto & targetPtgNode : pointsToGraph.getTargets(registerPtgNode).Items())
  {
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && pointsToGraph.isNodeConstant(targetPtgNode))
      continue;
    if (ENABLE_OPERATION_SIZE_BLOCKING && minTargetSize)
    {
      const auto targetSize = pointsToGraph.tryGetNodeSize(targetPtgNode);
      if (targetSize && *targetSize < minTargetSize)
        continue;
    }
    if (ENABLE_DEAD_ALLOCA_BLOCKLIST && allocasDead.Contains(targetPtgNode))
      continue;

    // We do not bother to check against adding doubled-up explicit loads/stores,
    // as the PointsToGraph should already be free of doubled-up targets.

    if constexpr (IsStore)
      modRefGraph.addExplicitStore(modRefNode, targetPtgNode);
    else
      modRefGraph.addExplicitLoad(modRefNode, targetPtgNode);
  }
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateLoad(
    const rvsdg::SimpleNode & loadNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = Context_->modRefGraph.getOrCreateModRefForNode(loadNode);
  const auto origin = LoadOperation::AddressInput(loadNode).origin();
  const auto loadOperation = util::assertedCast<const LoadOperation>(&loadNode.GetOperation());
  const auto loadSize = GetTypeSize(*loadOperation->GetLoadedType());

  addPointerOriginTargets<false>(nodeModRef, *origin, loadSize, lambda);
  return nodeModRef;
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateStore(
    const rvsdg::SimpleNode & storeNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = Context_->modRefGraph.getOrCreateModRefForNode(storeNode);
  const auto origin = StoreOperation::AddressInput(storeNode).origin();
  const auto storeOperation = util::assertedCast<const StoreOperation>(&storeNode.GetOperation());
  const auto storeSize = GetTypeSize(storeOperation->GetStoredType());

  addPointerOriginTargets<true>(nodeModRef, *origin, storeSize, lambda);
  return nodeModRef;
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateAlloca(const rvsdg::SimpleNode & allocaNode)
{
  const auto nodeModRef = Context_->modRefGraph.getOrCreateModRefForNode(allocaNode);
  const auto allocaPtgNode = Context_->pointsToGraph.getAllocaNode(allocaNode);

  // The creation of the alloca is considered to be a store
  Context_->modRefGraph.addExplicitStore(nodeModRef, allocaPtgNode);
  return nodeModRef;
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateMalloc(const rvsdg::SimpleNode & mallocNode)
{
  const auto nodeModRef = Context_->modRefGraph.getOrCreateModRefForNode(mallocNode);
  const auto & mallocPtgNode = Context_->pointsToGraph.getMallocNode(mallocNode);

  // The creation of the malloc is considered to be a store
  Context_->modRefGraph.addExplicitStore(nodeModRef, mallocPtgNode);
  return nodeModRef;
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateFree(
    const rvsdg::SimpleNode & freeNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<FreeOperation>(&freeNode));

  const auto nodeModRef = Context_->modRefGraph.getOrCreateModRefForNode(freeNode);
  const auto origin = freeNode.input(0)->origin();

  // Let the free operation behave as a store in its Mod/Ref set
  // TODO: Only free MallocMemoryNodes
  addPointerOriginTargets<true>(nodeModRef, *origin, std::nullopt, lambda);
  return nodeModRef;
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateMemcpy(
    const rvsdg::SimpleNode & memcpyNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<MemCpyOperation>(&memcpyNode));

  const auto nodeModRef = Context_->modRefGraph.getOrCreateModRefForNode(memcpyNode);
  const auto dstOrigin = MemCpyOperation::destinationInput(memcpyNode).origin();
  const auto srcOrigin = MemCpyOperation::sourceInput(memcpyNode).origin();
  const auto countOrigin = MemCpyOperation::countInput(memcpyNode).origin();
  auto count = tryGetConstantSignedInteger(*countOrigin);

  // Avoid underflow in signed -> unsigned conversion
  if (*count && count < 0)
    count = std::nullopt;

  addPointerOriginTargets<true>(nodeModRef, *dstOrigin, count, lambda);
  addPointerOriginTargets<false>(nodeModRef, *srcOrigin, count, lambda);
  return nodeModRef;
}

ModRefNodeIndex
RegionAwareModRefSummarizer::annotateCall(
    const rvsdg::SimpleNode & callNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  const auto & pointsToGraph = Context_->pointsToGraph;
  auto & modRefGraph = Context_->modRefGraph;

  // This ModRefNode represents everything the call may affect
  const auto callModRef = modRefGraph.getOrCreateModRefForNode(callNode);

  // Go over all possible targets of the call and add them to the call summary
  const auto targetPtr = callNode.input(0)->origin();
  const auto & targetPtgNode = Context_->pointsToGraph.getRegisterNode(*targetPtr);

  // Go through all locations the called function pointer may target
  for (auto & calleePtgNode : pointsToGraph.getTargets(targetPtgNode).Items())
  {
    const auto kind = pointsToGraph.getKind(calleePtgNode);
    if (kind == PointsToGraph::NodeKind::LambdaNode)
    {
      const auto & calleeLambda = pointsToGraph.getLambdaNodeObject(calleePtgNode);
      const auto calleeModRef = modRefGraph.getOrCreateModRefForNode(calleeLambda);
      modRefGraph.addSimpleConstraintEdge(calleeModRef, callModRef);
    }
    else if (kind == PointsToGraph::NodeKind::ImportNode)
    {
      modRefGraph.markAsCallingExternal(callModRef);
    }
  }
  if (pointsToGraph.isTargetingAllExternallyAvailable(targetPtgNode))
  {
    modRefGraph.markAsCallingExternal(callModRef);
  }

  // Allocas that are live within the call, might no longer be live from the call site
  if (ENABLE_DEAD_ALLOCA_BLOCKLIST)
  {
    modRefGraph.addBlocklist(
        callModRef,
        Context_->AllocasDeadInScc[Context_->FunctionToSccIndex[&lambda]]);
  }

  return callModRef;
}

void
RegionAwareModRefSummarizer::createExternalModRefNode()
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  Context_->externalModRefNode = Context_->modRefGraph.createModRefNode();
  // An external function may call another external function
  Context_->modRefGraph.markAsCallingExternal(Context_->externalModRefNode);

  // All functions that are externally available can be called from external functions
  for (auto lambdaPtgNode : pointsToGraph.lambdaNodes())
  {
    if (!pointsToGraph.isExternallyAvailable(lambdaPtgNode))
      continue;

    // Add a call from external to the function
    const auto & lambdaNode = pointsToGraph.getLambdaNodeObject(lambdaPtgNode);
    auto lambdaModRefIndex = Context_->modRefGraph.getOrCreateModRefForNode(lambdaNode);
    Context_->modRefGraph.addSimpleConstraintEdge(lambdaModRefIndex, Context_->externalModRefNode);
  }
}

void
RegionAwareModRefSummarizer::createMemoryNodeOrdering()
{
  const auto & pointsToGraph = Context_->pointsToGraph;
  const auto & modRefGraph = Context_->modRefGraph;

  // Add all relevant memory nodes to this list
  std::vector<PointsToGraph::NodeIndex> memoryNodeOrder;

  for (PointsToGraph::NodeIndex ptgNode = 0; ptgNode < pointsToGraph.numNodes(); ptgNode++)
  {
    if (!pointsToGraph.isMemoryNode(ptgNode))
      continue;

    if (ENABLE_CONSTANT_MEMORY_BLOCKING && pointsToGraph.isNodeConstant(ptgNode))
      continue;

    memoryNodeOrder.push_back(ptgNode);
  }

  // Sorting rules, in order of most important to least:
  // - All externally available memory comes first
  // Among nodes that are not externally available,
  // they are sorted by how they can be affected by external function calls:
  //   [ stored but not loaded ][ stored and loaded ][ loaded but not stored ][ neither ]
  // - Memory nodes are sorted by increasing size, with unknown sizes at the end
  // - Memory nodes are sorted by kind
  // - Alloca and malloc nodes are sorted by the function they belong to

  std::sort(
      memoryNodeOrder.begin(),
      memoryNodeOrder.end(),
      [&](PointsToGraph::NodeIndex a, PointsToGraph::NodeIndex b) -> bool
      {
        // The sort comparison function returns true if a should come before b

        const bool aIsExternallyAvailable = pointsToGraph.isExternallyAvailable(a);
        const bool bIsExternallyAvailable = pointsToGraph.isExternallyAvailable(b);
        if (aIsExternallyAvailable != bIsExternallyAvailable)
          return aIsExternallyAvailable;

        const bool bothPrivate = !aIsExternallyAvailable && !bIsExternallyAvailable;
        if (bothPrivate)
        {
          const auto & storedByExternal =
              modRefGraph.getExplicitStores(Context_->externalModRefNode);
          const bool aIsStoredByExternal = storedByExternal.Contains(a);
          const bool bIsStoredByExternal = storedByExternal.Contains(b);
          const auto & loadedByExternal =
              modRefGraph.getExplicitLoads(Context_->externalModRefNode);
          const bool aIsLoadedByExternal = loadedByExternal.Contains(a);
          const bool bIsLoadedByExternal = loadedByExternal.Contains(b);

          // Memory nodes that are both stored to and loaded from by external functions,
          // are placed in the middle. Memory nodes that are neither come last.
          const int aPriority = (aIsStoredByExternal * 3) ^ aIsLoadedByExternal;
          const int bPriority = (bIsStoredByExternal * 3) ^ bIsLoadedByExternal;
          if (aPriority != bPriority)
            return aPriority > bPriority;
        }

        const auto aSize = pointsToGraph.tryGetNodeSize(a);
        const auto bSize = pointsToGraph.tryGetNodeSize(b);
        if (aSize != bSize)
        {
          // Unknown sizes come last
          if (!aSize.has_value())
            return false;
          if (!bSize.has_value())
            return true;
          return *aSize < *bSize;
        }

        const auto aKind = pointsToGraph.getKind(a);
        const auto bKind = pointsToGraph.getKind(b);
        if (aKind != bKind)
          return aKind < bKind;

        if (aKind == PointsToGraph::NodeKind::AllocaNode)
        {
          const auto & aAllocaNode = pointsToGraph.getAllocaNodeObject(a);
          const auto & bAllocaNode = pointsToGraph.getAllocaNodeObject(b);
          const auto aLambdaId = getSurroundingLambdaNode(aAllocaNode).GetNodeId();
          const auto bLambdaId = getSurroundingLambdaNode(bAllocaNode).GetNodeId();
          if (aLambdaId != bLambdaId)
            return aLambdaId < bLambdaId;
        }
        if (aKind == PointsToGraph::NodeKind::MallocNode)
        {
          const auto & aMallocNode = pointsToGraph.getMallocNodeObject(a);
          const auto & bMallocNode = pointsToGraph.getMallocNodeObject(b);
          const auto aLambdaId = getSurroundingLambdaNode(aMallocNode).GetNodeId();
          const auto bLambdaId = getSurroundingLambdaNode(bMallocNode).GetNodeId();
          if (aLambdaId != bLambdaId)
            return aLambdaId < bLambdaId;
        }

        // Final comparison if all else is equal
        return a < b;
      });

  // Create metadata about the MemoryNodeOrdering
  auto & metadata = Context_->memoryNodeOrderingMetadata;
  metadata.ptgNodeIndexToMemoryOrderingIndex.resize(pointsToGraph.numNodes());

  // Go through the MemoryNodeOrdering in order and track relevant intervals
  std::optional<size_t> previousExternallyAvailableNodeSize = std::nullopt;
  std::optional<MemoryNodeOrderingIndex> firstStoredByExternal = std::nullopt;
  std::optional<MemoryNodeOrderingIndex> firstLoadedByExternal = std::nullopt;
  for (MemoryNodeOrderingIndex i = 0; i < memoryNodeOrder.size(); i++)
  {
    const auto ptgNode = memoryNodeOrder[i];
    metadata.ptgNodeIndexToMemoryOrderingIndex[ptgNode] = i;

    if (pointsToGraph.isExternallyAvailable(ptgNode))
    {
      metadata.endOfExternallyAvailable = i + 1;

      // nodes with unknown size are given a size larger than anything else,
      auto size =
          pointsToGraph.tryGetNodeSize(ptgNode).value_or(std::numeric_limits<size_t>::max());
      if (size != previousExternallyAvailableNodeSize)
      {
        metadata.firstExternallyAvailableWithSize[size] = i;
        previousExternallyAvailableNodeSize = size;
      }
    }
    else
    {
      const bool storedByExternal =
          modRefGraph.getExplicitStores(Context_->externalModRefNode).Contains(ptgNode);
      if (storedByExternal)
      {
        if (!firstStoredByExternal)
          firstStoredByExternal = i;
        metadata.endOfStoredByExternal = i + 1;
      }
      const bool loadedByExternal =
          modRefGraph.getExplicitLoads(Context_->externalModRefNode).Contains(ptgNode);
      if (loadedByExternal)
      {
        if (!firstLoadedByExternal)
          firstLoadedByExternal = i;
        metadata.endOfLoadedByExternal = i + 1;
      }
    }
  }

  metadata.startOfStoredByExternal = firstStoredByExternal.value_or(0);
  metadata.startOfLoadedByExternal = firstLoadedByExternal.value_or(0);

  Context_->memoryNodeOrdering =
      std::make_unique<MemoryNodeOrdering>(pointsToGraph, std::move(memoryNodeOrder));
}

std::unique_ptr<RegionAwareModRefSummarizer::RegionAwareModRefSummary>
RegionAwareModRefSummarizer::createModRefSummary()
{
  JLM_ASSERT(Context_->memoryNodeOrdering);

  const auto & modRefGraph = Context_->modRefGraph;
  const auto & memoryNodeOrdering = *Context_->memoryNodeOrdering;
  const auto & metadata = Context_->memoryNodeOrderingMetadata;

  std::vector<ModRefSet> modRefSets;

  // Convert each ModRefNode to a ModRefSet consisting of intervals
  for (ModRefNodeIndex i = 0; i < modRefGraph.numModRefNodes(); i++)
  {
    std::vector<MemoryNodeInterval> loadIntervals;
    std::vector<MemoryNodeInterval> storeIntervals;

    // Handle ModRefs that load from all external memory nodes with a size >= X
    const auto loadingFromExternal = modRefGraph.isLoadingFromExternal(i);
    if (loadingFromExternal)
    {
      // Returns the first MemoryNodeIndex with size >= loadingFromExternal
      const auto it = metadata.firstExternallyAvailableWithSize.lower_bound(*loadingFromExternal);
      if (it != metadata.firstExternallyAvailableWithSize.end())
      {
        loadIntervals.push_back(MemoryNodeInterval(it->second, metadata.endOfExternallyAvailable));
      }
    }

    // Handle ModRefs to that store to all external memory nodes with a size >= X
    const auto storingToExternal = modRefGraph.isStoringToExternal(i);
    if (storingToExternal)
    {
      // Returns the first MemoryNodeIndex with size >= storingToExternal
      const auto it = metadata.firstExternallyAvailableWithSize.lower_bound(*storingToExternal);
      if (it != metadata.firstExternallyAvailableWithSize.end())
      {
        storeIntervals.push_back(MemoryNodeInterval(it->second, metadata.endOfExternallyAvailable));
      }
    }

    // Handle ModRefNodes that make calls to external functions
    if (modRefGraph.isCallingExternal(i))
    {
      loadIntervals.push_back(
          MemoryNodeInterval(metadata.startOfLoadedByExternal, metadata.endOfLoadedByExternal));
      storeIntervals.push_back(
          MemoryNodeInterval(metadata.startOfStoredByExternal, metadata.endOfStoredByExternal));
    }

    // Handle explicit loads and stores
    for (const auto load : modRefGraph.getExplicitLoads(i).Items())
    {
      const auto memoryOrderingIndex = metadata.ptgNodeIndexToMemoryOrderingIndex[load];
      loadIntervals.push_back(MemoryNodeInterval(memoryOrderingIndex));
    }
    for (const auto store : modRefGraph.getExplicitStores(i).Items())
    {
      const auto memoryOrderingIndex = metadata.ptgNodeIndexToMemoryOrderingIndex[store];
      storeIntervals.push_back(MemoryNodeInterval(memoryOrderingIndex));
    }

    // The constructor of MemoryNodeIntervalSet fixes sorting, merging and removing empty intervals
    MemoryNodeIntervalSet loads(std::move(loadIntervals));
    MemoryNodeIntervalSet stores(std::move(storeIntervals));

    modRefSets.push_back(ModRefSet(std::move(loads), std::move(stores)));
  }

  auto result = std::make_unique<RegionAwareModRefSummary>(
      std::move(memoryNodeOrdering),
      std::move(modRefSets),
      modRefGraph.getNodeMap());

  return result;
}

std::string
RegionAwareModRefSummarizer::callGraphSCCsToString() const
{
  std::ostringstream ss;
  for (size_t i = 0; i < Context_->SccFunctions.size(); i++)
  {
    if (i != 0)
      ss << " <- ";
    ss << "[" << std::endl;
    if (i == Context_->ExternalNodeSccIndex)
    {
      ss << "  " << "<external>" << std::endl;
    }
    for (auto function : Context_->SccFunctions[i].Items())
    {
      ss << "  " << function->DebugString() << std::endl;
    }
    ss << "]";
  }
  return ss.str();
}

std::string
RegionAwareModRefSummarizer::dumpRegionTree(const rvsdg::Graph & rvsdg)
{
  const auto & modRefGraph = Context_->modRefGraph;

  std::ostringstream ss;

  auto indent = [&](size_t depth, char c = '-')
  {
    for (size_t i = 0; i < depth; i++)
      ss << c;
  };

  std::function<void(const rvsdg::Node &, size_t)> toRegionTree =
      [&](const rvsdg::Node & node, size_t depth)
  {
    if (!modRefGraph.hasModRefForNode(node))
      return;

    indent(depth, ' ');
    ss << node.DebugString() << ": (" << &node << ") ";
    auto modRefIndex = modRefGraph.getModRefForNode(node);
    ss << modRefGraph.debugStringForSet(modRefIndex) << std::endl;

    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & region : structuralNode->Subregions())
      {
        indent(depth + 1, '-');
        ss << "region" << std::endl;
        for (auto & n : region.Nodes())
          toRegionTree(n, depth + 2);
      }
    }
  };

  for (auto & node : rvsdg.GetRootRegion().Nodes())
    toRegionTree(node, 0);

  return ss.str();
}

std::unique_ptr<ModRefSummary>
RegionAwareModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  RegionAwareModRefSummarizer summarizer;
  return summarizer.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<ModRefSummary>
RegionAwareModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph)
{
  util::StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

}
