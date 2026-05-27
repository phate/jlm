/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/rvsdg/lambda.hpp"
#include "jlm/util/common.hpp"
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/TarjanScc.hpp>
#include <jlm/util/Worklist.hpp>

#include <limits>
#include <optional>
#include <queue>
#include <unordered_map>

namespace jlm::llvm::aa
{

/**
 * allocas that are not defined in f(), and not defined in a predecessor of f() in the call graph,
 * cannot be live inside f(). They are added to the DeadAllocaBlocklist.
 */
static const bool ENABLE_DEAD_ALLOCA_BLOCKLIST = !std::getenv("JLM_DISABLE_DEAD_ALLOCA_BLOCKLIST");

/**
 * In a region with an alloca definition, the memory node representing the alloca does not need to
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
 * Within each function, memory nodes that always occur together with the extern node,
 * can be removed from all mod ref sets.
 */
static const bool ENABLE_EXTERNAL_COMPRESSION = !std::getenv("JLM_DISABLE_EXTERN_COMPRESSION");

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
  static constexpr auto NumFunctionsCallingSetjmp_ = "#FunctionsCallingSetjmp";
  static constexpr auto NumCallGraphSccsCanCallExternal_ = "#CallGraphSccsCanCallExternal";

  // Statistics about compressing memory nodes into the external node
  // Number of ModRefSets containing external
  static constexpr auto NumExtModRefSets_ = "#ExtModRefSets";
  // Number of memory nodes removed from compression in sets containing external
  static constexpr auto NumExtModRefCompressed_ = "#ExtModRefCompressed";
  // Number of memory nodes kept (not compressed) in sets containing external
  static constexpr auto NumExtModRefKept_ = "#ExtModRefKept";
  // Number of ModRefSets that do not contain external
  static constexpr auto NumLocalModRefSets_ = "#LocalModRefSets";
  // Number of memory nodes in sets that do not contain external
  static constexpr auto NumLocalModRefKept_ = "#LocalModRefKept";

  static constexpr auto CallGraphTimer_ = "CallGraphTimer";
  static constexpr auto AllocasDeadInSccsTimer_ = "AllocasDeadInSccsTimer";
  static constexpr auto SimpleAllocasSetTimer_ = "SimpleAllocasSetTimer";
  static constexpr auto NonReentrantAllocaSetsTimer_ = "NonReentrantAllocaSetsTimer";
  static constexpr auto CreateExternalModRefSetTimer_ = "CreateExternalModRefSetTimer";
  static constexpr auto AnnotationTimer_ = "AnnotationTimer";
  static constexpr auto SolvingTimer_ = "SolvingTimer";
  static constexpr auto ExternalCompressionTimer_ = "ExternalCompactionTimer";

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
  startCallGraphStatistics()
  {
    AddTimer(CallGraphTimer_).start();
  }

  void
  stopCallGraphStatistics(size_t numSccs, size_t numFunctionsCallingSetjmp)
  {
    GetTimer(CallGraphTimer_).stop();
    AddMeasurement(NumCallGraphSccs_, numSccs);
    AddMeasurement(NumFunctionsCallingSetjmp_, numFunctionsCallingSetjmp);
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
  StartCreateExternalModRefSet()
  {
    AddTimer(CreateExternalModRefSetTimer_).start();
  }

  void
  StopCreateExternalModRefSet()
  {
    GetTimer(CreateExternalModRefSetTimer_).stop();
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
  StartExternalCompressionStatistics()
  {
    AddTimer(ExternalCompressionTimer_).start();
  }

  void
  StopExternalCompressionStatistics(
      size_t numExtModRefSets,
      size_t numExtModRefCompressed,
      size_t numExtModRefKept,
      size_t numLocalModRefSets,
      size_t numLocalModRefKept)
  {
    GetTimer(ExternalCompressionTimer_).stop();
    AddMeasurement(NumExtModRefSets_, numExtModRefSets);
    AddMeasurement(NumExtModRefCompressed_, numExtModRefCompressed);
    AddMeasurement(NumExtModRefKept_, numExtModRefKept);
    AddMeasurement(NumLocalModRefSets_, numLocalModRefSets);
    AddMeasurement(NumLocalModRefKept_, numLocalModRefKept);
  }

  static std::unique_ptr<Statistics>
  Create(const rvsdg::RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Statistics>(rvsdgModule, pointsToGraph);
  }
};

/**
 * ModRefSet subclass used by RegionAwareModRefSummarizer.
 *
 * The class uses both implicit and explicit representations of memory nodes.
 * The implicit representation is used for accesses to externally available memory,
 * and calls (or structural nodes that contain calls) to functions in external modules.
 * The explicit representation uses hash maps containing individual memory nodes.
 *
 * After solving, implicit memory nodes are materialized into the explicit sets,
 * with the exception of memory nodes that can be compressed into the external memory node.
 * Constant memory is also left out of the final materialized sets.
 *
 * @see ModRefSet
 */
class RegionAwareModRefSet final : public ModRefSet
{
public:
  // The byte size used when no access to external is made
  static constexpr uint32_t NoneSize = std::numeric_limits<uint32_t>::max();

  RegionAwareModRefSet() = default;

  /**
   * Checks if an externally available memory node with the given size may be read by this set.
   * @param memoryNodeSize the size of the externally available memory node, or nullopt if unknown.
   * @return true if this set represents possibly referencing the given memory node
   */
  [[nodiscard]] bool
  mayRefExternalOfSize(std::optional<size_t> memoryNodeSize)
  {
    if (refExternalOfSize_ == NoneSize)
      return false;
    if (!memoryNodeSize.has_value())
      return true;
    return *memoryNodeSize >= refExternalOfSize_;
  }

  /**
   * Checks if an externally available memory node with the given size may be modified by this set.
   * @param memoryNodeSize the size of the externally available memory node, or nullopt if unknown.
   * @return true if this set represents possibly modifying the given memory node
   */
  [[nodiscard]] bool
  mayModExternalOfSize(std::optional<size_t> memoryNodeSize)
  {
    if (modExternalOfSize_ == NoneSize)
      return false;
    if (!memoryNodeSize.has_value())
      return true;
    return *memoryNodeSize >= modExternalOfSize_;
  }

  /**
   * Marks the ModRefSet as possibly referencing all externally available memory of size >= minSize
   * @return true if the ModRefSet was modified by this operation, otherwise false
   */
  bool
  markAsLoadingFromExternal(uint32_t minSize)
  {
    if (minSize < refExternalOfSize_)
    {
      refExternalOfSize_ = minSize;
      return true;
    }
    return false;
  }

  /**
   * Marks the ModRefSet as possibly referencing, and/or modifying
   * all externally available memory of size >= minSize.
   * @return true if the ModRefSet was modified by this operation, otherwise false
   */
  bool
  markAsStoringToExternal(uint32_t minSize)
  {
    if (minSize < modExternalOfSize_)
    {
      // refExternalOfSize is always <= modExternalOfSize
      refExternalOfSize_ = minSize;
      modExternalOfSize_ = minSize;
      return true;
    }
    return false;
  }

  /**
   * @return true if the ModRefSet may represent a call to an externally defined function.
   */
  [[nodiscard]]
  bool mayCallExternalFunction()
  {
    return callsExternalFunction_;
  }

  /**
   * Marks the ModRefSet as possibly containing calls to externally defined functions,
   * as represented by the shared ExternalModRefIndex in the summarizer.
   * @return true if the ModRefSet was modified by this operation, otherwise false
   */
  bool
  markAsCallingExternalFunction()
  {
    if (callsExternalFunction_)
      return false;

    callsExternalFunction_ = true;
    refExternalOfSize_ = 0;
    modExternalOfSize_ = 0;
    return true;
  }

  /**
   * Adds the given memory node as an explicit member of the ModRefSet.
   * @param memoryNode the index of the memory node in the points-to graph.
   * @param mayMod if true, the memory node may be modified. If false it may only be referenced.
   * @return true if the ModRefSet was modified by this operation, otherwise false
   */
  bool
  addExplicitMemoryNode(PointsToGraph::NodeIndex memoryNode, bool mayMod)
  {
    const auto [it, inserted] = modRefNodes_.insert({memoryNode, mayMod});
    if (inserted)
      return true;

    // The memory node was already present, but we may have updated a ref to a mod/ref.
    if (!it->second && mayMod)
    {
      it->second = true;
      return true;
    }
    return false;
  }

  /**
   * Propagates all flags from the given set \p other to this set.
   * @return true if any flags in this ModRefSet changed, false otherwise
   */
  bool
  propagateFlags(const RegionAwareModRefSet & other)
  {
    bool changed = false;
    if (other.refExternalOfSize_ < refExternalOfSize_)
    {
      refExternalOfSize_ = other.refExternalOfSize_;
      changed |= true;
    }
    if (other.modExternalOfSize_ < modExternalOfSize_)
    {
      modExternalOfSize_ = other.modExternalOfSize_;
      changed |= true;
    }
    if (other.callsExternalFunction_ && !callsExternalFunction_)
    {
      callsExternalFunction_ = other.callsExternalFunction_;
      changed |= true;
    }

    return changed;
  }

private:

  // If not equal to NoneSize, the ModRefSet potentially reads all externally available memory
  // of size >= the given number of bytes.
  uint32_t refExternalOfSize_ = NoneSize;
  // If not equal to NoneSize, the ModRefSet potentially modifies all externally available memory
  // of size >= the given number of bytes.
  // Must be greater than or equal to \ref refExternalOfSize_
  uint32_t modExternalOfSize_ = NoneSize;
  // If true, the ModRefSet represents possibly calling externally defined functions.
  // Implies that \ref refExternalOfSize and \ref modExternalOfSize are 0
  bool callsExternalFunction_ = false;
};

/** \brief Mod/Ref summary of region-aware mod/ref summarizer
 */
class RegionAwareModRefSummary final : public ModRefSummary
{
public:
  explicit RegionAwareModRefSummary(const PointsToGraph & pointsToGraph)
      : pointsToGraph_(pointsToGraph)
  {
    // Always create a ModRefSet representing external functions
    externModRefSet_ = createModRefSet();
    // The ModRefSet representing external functions can call external functions
    markSetAsCallingExternalFunction(externModRefSet_);
  }

  RegionAwareModRefSummary(const RegionAwareModRefSummary &) = delete;
  RegionAwareModRefSummary &
  operator=(const RegionAwareModRefSummary &) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return pointsToGraph_;
  }

  [[nodiscard]] size_t
  NumModRefSets() const noexcept
  {
    return modRefSets_.size();
  }

  [[nodiscard]] RegionAwareModRefSet &
  getModRefSet(ModRefSetIndex index)
  {
    JLM_ASSERT(index < modRefSets_.size());
    return modRefSets_[index];
  }

  [[nodiscard]] const RegionAwareModRefSet &
  getModRefSet(ModRefSetIndex index) const
  {
    JLM_ASSERT(index < modRefSets_.size());
    return modRefSets_[index];
  }

  bool markSetAsLoadingFromExternal(ModRefSetIndex index, size_t minSize)
  {
    return getModRefSet(index).markAsLoadingFromExternal(minSize);
  }

  bool markSetAsStoringToExternal(ModRefSetIndex index, size_t minSize)
  {
    return getModRefSet(index).markAsStoringToExternal(minSize);
  }

  bool markSetAsCallingExternalFunction(ModRefSetIndex index)
  {
    return getModRefSet(index).markAsCallingExternalFunction();
  }

  /**
   * Adds the given \ref ptgNode the ModRefSet with the given \p index.
   * The parameter \p mayMod indicates if the operation is only a reference,
   * or possibly also modifies the memory represented by the memory node.
   *
   * @return true if the ModRefSet grew, false otherwise.
   */
  bool
  addMemoryNodeToSet(ModRefSetIndex index, PointsToGraph::NodeIndex ptgNode, bool mayMod)
  {
    // The external memory node should not be added explicitly until solving is finished
    // and implicit members get materialized
    JLM_ASSERT(ptgNode != PointsToGraph::externalMemoryNode);
    auto & modRefSet = getModRefSet(index);

    // If the memory nodes is already implicitly included in the ModRefSet, avoid adding it
    if (pointsToGraph_.isExternallyAvailable(ptgNode))
    {
      const auto memoryNodeSize = pointsToGraph_.tryGetNodeSize(ptgNode);
      if (mayMod)
      {
        if (modRefSet.mayModExternalOfSize(memoryNodeSize))
          return false;
      }
      else
      {
        if (modRefSet.mayRefExternalOfSize(memoryNodeSize))
          return false;
      }
    }

    return modRefSet.addExplicitMemoryNode(ptgNode, mayMod);
  }

  /**
   * Makes the \p to ModRefSet contain a superset of the memory nodes in the \p from ModRefSet.
   * Propagates both flags and explicit memory nodes.
   * @return true if this operation modified the to set, false otherwise.
   */
  bool
  propagateModRefSet(ModRefSetIndex from, ModRefSetIndex to)
  {
    JLM_ASSERT(from < modRefSets_.size());
    JLM_ASSERT(to < modRefSets_.size());
    const auto & fromSet = modRefSets_[from];
    auto & toSet = modRefSets_[to];

    // Propagate flags first, as they may allow explicit memory nodes to be skipped
    bool changed = toSet.propagateFlags(fromSet);
    // Propagate all explicit memory nodes as well
    for (auto [memoryNode, mayMod] : fromSet.getModRefNodes())
    {
      changed |= addMemoryNodeToSet(to, memoryNode, mayMod);
    }
    return changed;
  }

  [[nodiscard]] ModRefSetIndex
  getExternModRefSet()
  {
    return externModRefSet_;
  }

  [[nodiscard]] bool
  hasSetForNode(const rvsdg::Node & node) const
  {
    return nodeMap_.find(&node) != nodeMap_.end();
  }

  [[nodiscard]] ModRefSetIndex
  getSetForNode(const rvsdg::Node & node) const
  {
    const auto it = nodeMap_.find(&node);
    JLM_ASSERT(it != nodeMap_.end());
    return it->second;
  }

  /**
   * Get the ModRefSet associated with the given \p node, or creates one if none exists.
   * @param node the RVSDG node to be represented by a ModRefSet
   * @param lambdaNode the function the RVSDG node belongs to.
   */
  [[nodiscard]] ModRefSetIndex
  getOrCreateSetForNode(const rvsdg::Node & node, rvsdg::LambdaNode & lambdaNode)
  {
    auto [it, inserted] = nodeMap_.insert({&node, 0});
    if (inserted)
    {
      const auto created = createModRefSet();
      it->second = created;
      modRefSetsInFunction_[&lambdaNode].push_back(created);
      return created;
    }

    return it->second;
  }

  const ModRefSet &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const override
  {
    return modRefSets_[getSetForNode(node)];
  }

  const ModRefSet &
  GetGammaEntryModRef(const rvsdg::GammaNode & gamma) const override
  {
    return modRefSets_[getSetForNode(gamma)];
  }

  const ModRefSet &
  GetGammaExitModRef(const rvsdg::GammaNode & gamma) const override
  {
    return GetGammaEntryModRef(gamma);
  }

  const ModRefSet &
  GetThetaModRef(const rvsdg::ThetaNode & theta) const override
  {
    return modRefSets_[getSetForNode(theta)];
  }

  const ModRefSet &
  GetLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const override
  {
    return modRefSets_[getSetForNode(lambda)];
  }

  const ModRefSet &
  GetLambdaExitModRef(const rvsdg::LambdaNode & lambda) const override
  {
    return GetLambdaEntryModRef(lambda);
  }

  [[nodiscard]] static std::unique_ptr<RegionAwareModRefSummary>
  Create(const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<RegionAwareModRefSummary>(pointsToGraph);
  }

private:

  [[nodiscard]]
  ModRefSetIndex createModRefSet()
  {
    modRefSets_.emplace_back();
    return modRefSets_.size() - 1;
  }

  const PointsToGraph & pointsToGraph_;

  /**
   * All sets of ModRef information in the summary
   */
  std::vector<RegionAwareModRefSet> modRefSets_;

  /**
   * The ModRefSet representing everything that can be referenced/modified by external functions.
   * This includes from making calls to functions defined in the current module.
   */
  ModRefSetIndex externModRefSet_;

  /**
   * Lists ModRefSets grouped by the function of node they belong to.
   */
  std::unordered_map<const rvsdg::LambdaNode *, std::vector<ModRefSetIndex>> modRefSetsInFunction_;

  /**
   * Map from nodes that have memory side effects, to their ModRefSet.
   * Includes nodes like loads, stores, memcpy, free and calls.
   * Also includes structural nodes like gamma, theta and lambda.
   */
  std::unordered_map<const rvsdg::Node *, ModRefSetIndex> nodeMap_;
};

/**
 * Struct holding temporary data used during the creation of a single mod/ref summary
 */
struct RegionAwareModRefSummarizer::Context
{
  explicit Context(const PointsToGraph & ptg)
      : pointsToGraph(ptg)
  {}

  /**
   * The points to graph used to create the Mod/Ref summary.
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
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  std::vector<util::HashSet<const rvsdg::LambdaNode *>> SccFunctions;

  /**
   * The index of the SCC in the call graph that represent containing all external functions
   *
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  size_t ExternalNodeSccIndex = 0;

  /**
   * For each SCC in the call graph, the set of SCCs it targets using calls.
   * Since SCCs are ordered in reverse topological order, an SCC never targets higher indices.
   * If there is any possibility of recursion within an SCC, it also targets itself.
   *
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  std::vector<util::HashSet<size_t>> SccCallTargets;

  /**
   * A mapping from functions to the index of the SCC they belong to in the call graph
   *
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  std::unordered_map<const rvsdg::LambdaNode *, size_t> FunctionToSccIndex;

  /**
   * The set of functions that call setjmp directly.
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  util::HashSet<const rvsdg::LambdaNode *> FunctionsCallingSetjmp;

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
  util::HashSet<PointsToGraph::NodeIndex> SimpleAllocas;

  /**
   * For each region, this field contains the set of allocas defined in the region,
   * that have been shown to be non-reentrant.
   *
   * Assigned in \ref CreateNonReentrantAllocaSets(). Remains constant after.
   */
  std::unordered_map<const rvsdg::Region *, util::HashSet<PointsToGraph::NodeIndex>>
      NonReentrantAllocas;

  /**
   * Simple edges in the ModRefSet constraint graph.
   * A simple edge a -> b indicates that the ModRefSet b should contain everything in a.
   * ModRefSetSimpleEdges[a] contains b, as well as any other simple edge successors.
   */
  std::vector<util::HashSet<ModRefSetIndex>> ModRefSetSimpleConstraints;

  /**
   * Blocklists in the ModRefSet constraint graphs.
   * During solving, a MemoryNode x that is about to be propagated to a ModRefSet a
   * will be skipped if x is in the Blocklist associated with a.
   * The pointer to the blocklist must remain valid until solving is finished.
   */
  std::unordered_map<ModRefSetIndex, const util::HashSet<PointsToGraph::NodeIndex> *>
      ModRefSetBlocklists;

  // Counters used for external compression statistics
  size_t numExtModRefSets = 0;
  size_t numExtModRefCompressed = 0;
  size_t numExtModRefKept = 0;
  size_t numLocalModRefSets = 0;
  size_t numLocalModRefKept = 0;
};

RegionAwareModRefSummarizer::~RegionAwareModRefSummarizer() noexcept = default;

RegionAwareModRefSummarizer::RegionAwareModRefSummarizer() = default;

std::unique_ptr<ModRefSummary>
RegionAwareModRefSummarizer::SummarizeModRefs(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  ModRefSummary_ = RegionAwareModRefSummary::Create(pointsToGraph);
  Context_ = std::make_unique<Context>(pointsToGraph);
  auto statistics = Statistics::Create(rvsdgModule, pointsToGraph);

  statistics->startCallGraphStatistics();
  createCallGraph(rvsdgModule);
  statistics->stopCallGraphStatistics(
      Context_->SccFunctions.size(),
      Context_->FunctionsCallingSetjmp.Size());

  statistics->StartAllocasDeadInSccStatistics();
  FindAllocasDeadInSccs();
  statistics->StopAllocasDeadInSccStatistics();

  statistics->StartCreateSimpleAllocasSetStatistics();
  Context_->SimpleAllocas = CreateSimpleAllocaSet(pointsToGraph);
  statistics->StopCreateSimpleAllocasSetStatistics(Context_->SimpleAllocas.Size());

  statistics->StartCreateNonReentrantAllocaSetsStatistics();
  auto numNonReentrantAllocas = CreateNonReentrantAllocaSets();
  statistics->StopCreateNonReentrantAllocaSetsStatistics(numNonReentrantAllocas);

  statistics->StartCreateExternalModRefSet();
  CreateExternalModRefSet();
  statistics->StopCreateExternalModRefSet();

  statistics->StartAnnotationStatistics();
  // Go through and recursively annotate all functions, regions and nodes
  for (const auto & scc : Context_->SccFunctions)
  {
    for (const auto lambda : scc.Items())
    {
      AnnotateFunction(*lambda);
    }
  }
  statistics->StopAnnotationStatistics();

  statistics->StartSolvingStatistics();
  SolveModRefSetConstraintGraph();
  statistics->StopSolvingStatistics();

  if (ENABLE_EXTERNAL_COMPRESSION)
  {
    statistics->StartExternalCompressionStatistics();
    doExternalCompression();
    statistics->StopExternalCompressionStatistics(
        Context_->numExtModRefSets,
        Context_->numExtModRefCompressed,
        Context_->numExtModRefKept,
        Context_->numLocalModRefSets,
        Context_->numLocalModRefKept);
  }

  // Print debug output
  // std::cerr << PointsToGraph::dumpDot(pointsToGraph) << std::endl;
  // std::cerr << "numSimpleAllocas: " << Context_->SimpleAllocas.Size() << std::endl;
  // std::cerr << "numNonReentrantAllocas: " << numNonReentrantAllocas << std::endl;
  // std::cerr << "Call Graph SCCs:" << std::endl << CallGraphSCCsToString(*this) << std::endl;
  // std::cerr << "RegionTree:" << std::endl
  //           << ToRegionTree(rvsdgModule.Rvsdg(), *ModRefSummary_) << std::endl;

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
  Context_.reset();
  return std::move(ModRefSummary_);
}

/**
 * Collects all lambda nodes defined in the given module, in an unspecified order.
 * @param rvsdgModule the module
 * @return a list of all lambda nodes in the module
 */
static std::vector<const rvsdg::LambdaNode *>
CollectLambdaNodes(const rvsdg::RvsdgModule & rvsdgModule)
{
  std::vector<const rvsdg::LambdaNode *> result;

  // Recursively traverses all structural nodes, but does not enter into lambdas
  const std::function<void(rvsdg::Region &)> CollectLambdasInRegion =
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
          CollectLambdasInRegion(*structural->subregion(i));
        }
      }
    }
  };

  CollectLambdasInRegion(rvsdgModule.Rvsdg().GetRootRegion());

  return result;
}

void
RegionAwareModRefSummarizer::createCallGraph(const rvsdg::RvsdgModule & rvsdgModule)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // The list of lambdas becomes the list of nodes in the call graph
  auto lambdaNodes = CollectLambdaNodes(rvsdgModule);

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

  // Outgoing edges for each node in the call graph, indexed by position in lambdaNodes
  std::vector<util::HashSet<size_t>> callGraphSuccessors(numCallGraphNodes);

  // Add outgoing edges from the given caller to any function the call may target
  const auto handleCall = [&](const rvsdg::SimpleNode & callNode, size_t callerIndex) -> void
  {
    const auto classification = CallOperation::ClassifyCall(callNode);
    if (classification->isSetjmpCall())
    {
      Context_->FunctionsCallingSetjmp.insert(lambdaNodes[callerIndex]);
      return;
    }

    const auto target = callNode.input(0)->origin();
    const auto targetPtgNode = pointsToGraph.getNodeForRegister(*target);

    // Go through all locations the called function pointer may target
    for (const auto calleePtgNode : pointsToGraph.getExplicitTargets(targetPtgNode).Items())
    {
      const auto kind = pointsToGraph.getNodeKind(calleePtgNode);
      if (kind == PointsToGraph::NodeKind::LambdaNode)
      {
        const auto & lambdaNode = pointsToGraph.getLambdaForNode(calleePtgNode);

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

    if (pointsToGraph.isTargetingAllExternallyAvailable(targetPtgNode))
    {
      // If the call target pointer is flagged, add an edge to external functions
      callGraphSuccessors[callerIndex].insert(externalNodeIndex);
    }
  };

  // Recursive function finding all call operations, adding edges to the call graph
  const std::function<void(const rvsdg::Region &, size_t)> handleCalls =
      [&](const rvsdg::Region & region, size_t callerIndex) -> void
  {
    for (auto & node : region.Nodes())
    {
      if (const auto [callNode, callOp] = rvsdg::TryGetSimpleNodeAndOptionalOp<CallOperation>(node);
          callOp)
      {
        handleCall(*callNode, callerIndex);
      }

      rvsdg::MatchType(
          node,
          [&](const rvsdg::StructuralNode & structural)
          {
            for (auto & subregion : structural.Subregions())
            {
              handleCalls(subregion, callerIndex);
            }
          });
    }
  };

  // For all functions, visit all their calls and add outgoing edges in the call graph
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    handleCalls(*lambdaNodes[i]->subregion(), i);

    // If the function has escaped, add an edge from the node representing all external functions
    if (pointsToGraph.isExternallyAvailable(pointsToGraph.getNodeForLambda(*lambdaNodes[i])))
    {
      callGraphSuccessors[externalNodeIndex].insert(i);
    }
  }

  // Finally, add the fact that the external node may call itself
  callGraphSuccessors[externalNodeIndex].insert(externalNodeIndex);

  // Used by the implementation of Tarjan's SCC algorithm
  const auto getSuccessors = [&](size_t nodeIndex)
  {
    return callGraphSuccessors[nodeIndex].Items();
  };

  // Find SCCs in the call graph
  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSCCs = util::FindStronglyConnectedComponents<size_t>(
      numCallGraphNodes,
      getSuccessors,
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

void
RegionAwareModRefSummarizer::FindAllocasDeadInSccs()
{
  // TODO: This is a candiate for removal.
  // Enabling or disabling it does nothing for number of loads removed
  const auto & pointsToGraph = Context_->pointsToGraph;

  // First find which allocas may be live in each SCC
  std::vector<util::HashSet<PointsToGraph::NodeIndex>> liveAllocas(Context_->SccFunctions.size());

  util::HashSet<PointsToGraph::NodeIndex> allAllocas;

  // Add all Allocas to the SCC of the function they are defined in
  for (auto allocaPtgNode : Context_->pointsToGraph.allocaNodes())
  {
    allAllocas.insert(allocaPtgNode);
    const auto & allocaNode = pointsToGraph.getAllocaForNode(allocaPtgNode);
    const auto & lambdaNode = rvsdg::getSurroundingLambdaNode(allocaNode);
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
RegionAwareModRefSummarizer::CreateSimpleAllocaSet(const PointsToGraph & pointsToGraph)
{
  // The set of allocas that are simple. Starts off as an over-approximation
  util::HashSet<PointsToGraph::NodeIndex> simpleAllocas;
  // A queue used to visit all PtG memory nodes that are not simple allocas
  std::queue<PointsToGraph::NodeIndex> notSimple;

  for (PointsToGraph::NodeIndex ptgNode = 0; ptgNode < pointsToGraph.numNodes(); ptgNode++)
  {
    // Only memory nodes are relevant
    if (!pointsToGraph.isMemoryNode(ptgNode))
      continue;

    // Allocas that are not externally available start of as presumed simple
    if (pointsToGraph.getNodeKind(ptgNode) == PointsToGraph::NodeKind::AllocaNode
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
    for (const auto targetPtgNode : pointsToGraph.getExplicitTargets(ptgNode).Items())
    {
      // If the target is currently in the simple allocas candiate set, move it to the queue
      if (simpleAllocas.Remove(targetPtgNode))
        notSimple.push(targetPtgNode);
    }
  }

  return simpleAllocas;
}

util::HashSet<PointsToGraph::NodeIndex>
RegionAwareModRefSummarizer::GetSimpleAllocasReachableFromRegionArguments(
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
    const auto ptgNode = pointsToGraph.getNodeForRegister(*argument);
    nodes.push(ptgNode);
  }

  // Traverse along PointsToGraph edges to find all reachable simple allocas
  while (!nodes.empty())
  {
    const auto ptgNode = nodes.front();
    nodes.pop();

    for (const auto targetPtgNode : pointsToGraph.getExplicitTargets(ptgNode).Items())
    {
      // We only are about following simple allocas, as simple allocas are only reachable from them.
      if (!Context_->SimpleAllocas.Contains(targetPtgNode))
        continue;

      if (reachableSimpleAllocas.insert(targetPtgNode))
        nodes.push(targetPtgNode);
    }
  }

  return reachableSimpleAllocas;
}

bool
RegionAwareModRefSummarizer::IsRecursionPossible(const rvsdg::LambdaNode & lambda) const
{
  const auto scc = Context_->FunctionToSccIndex[&lambda];
  return Context_->SccCallTargets[scc].Contains(scc);
}

size_t
RegionAwareModRefSummarizer::CreateNonReentrantAllocaSets()
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // Caching the sets of simple allocas reachable from region arguments
  std::unordered_map<const rvsdg::Region *, util::HashSet<PointsToGraph::NodeIndex>>
      reachableSimpleAllocas;

  // Returns the set of simple allocas reachable from the region's arguments
  const auto getReachableSimpleAllocas =
      [&](const rvsdg::Region & region) -> const util::HashSet<PointsToGraph::NodeIndex> &
  {
    if (const auto it = reachableSimpleAllocas.find(&region); it != reachableSimpleAllocas.end())
    {
      return it->second;
    }
    return reachableSimpleAllocas[&region] = GetSimpleAllocasReachableFromRegionArguments(region);
  };

  // Checks if the simple alloca represented by the given points-to graph node is non-reentrant
  const auto isNonReentrant = [&](PointsToGraph::NodeIndex simpleAllocaPtgNode) -> bool
  {
    auto & allocaNode = pointsToGraph.getAllocaForNode(simpleAllocaPtgNode);
    const auto & region = *allocaNode.region();

    // If the alloca's function is never involved in any recursion,
    // the alloca is trivially non-reentrant.
    const auto & lambda = getSurroundingLambdaNode(allocaNode);
    if (!IsRecursionPossible(lambda))
      return true;

    // In lambdas where recursion is possible, simple allocas that are reachable from
    // region arguments via edges in the points-to graph must be considered reentrant.
    if (getReachableSimpleAllocas(region).Contains(simpleAllocaPtgNode))
      return false;

    // Otherwise the simple alloca is non-reentrant
    return true;
  };

  size_t numNonReentrantAllocas = 0;

  // Only simple allocas are candidates for being non-reentrant
  for (auto simpleAllocaPtgNode : Context_->SimpleAllocas.Items())
  {
    if (isNonReentrant(simpleAllocaPtgNode))
    {
      const auto region = pointsToGraph.getAllocaForNode(simpleAllocaPtgNode).region();
      // Creates a set for the region if it does not already have one, and add the alloca
      Context_->NonReentrantAllocas[region].insert(simpleAllocaPtgNode);
      numNonReentrantAllocas++;
    }
  }

  return numNonReentrantAllocas;
}

void
RegionAwareModRefSummarizer::AddModRefSimpleConstraint(ModRefSetIndex from, ModRefSetIndex to)
{
  // Ensure the constraint vector is large enough
  // TODO: Fix performance!!
  Context_->ModRefSetSimpleConstraints.resize(ModRefSummary_->NumModRefSets());
  Context_->ModRefSetSimpleConstraints[from].insert(to);
}

void
RegionAwareModRefSummarizer::AddModRefSetBlocklist(
    ModRefSetIndex index,
    const util::HashSet<PointsToGraph::NodeIndex> & blocklist)
{
  JLM_ASSERT(Context_->ModRefSetBlocklists.find(index) == Context_->ModRefSetBlocklists.end());
  Context_->ModRefSetBlocklists[index] = &blocklist;
}

void
RegionAwareModRefSummarizer::AnnotateFunction(const rvsdg::LambdaNode & lambda)
{
  const auto & region = *lambda.subregion();
  const auto regionModRefSet = AnnotateRegion(region, lambda);
  const auto lambdaModRefSet = ModRefSummary_->getOrCreateSetForNode(lambda);
  AddModRefSimpleConstraint(regionModRefSet, lambdaModRefSet);

  if (Context_->FunctionsCallingSetjmp.Contains(&lambda))
  {
    // If this function can be jumped into, store operations on memory in its Mod/Ref set must be
    // sequentialized with calls to external functions, in case the trigger jumps
    // TODO: When we separate loads and stores, this edge should only propagate stores,
    // and turn them into loads
    AddModRefSimpleConstraint(lambdaModRefSet, ModRefSummary_->getExternModRefSet());
  }

  // If the function is externally available, it can be called by external functions,
  // so add a simple edge to the ModRefSet representing all external functions.
  const auto lambdaPtgNode = Context_->pointsToGraph.getNodeForLambda(lambda);
  if (Context_->pointsToGraph.isExternallyAvailable(lambdaPtgNode))
  {
    AddModRefSimpleConstraint(lambdaModRefSet, ModRefSummary_->getExternModRefSet());
  }
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateRegion(
    const rvsdg::Region & region,
    const rvsdg::LambdaNode & lambda)
{
  const auto regionModRefSet = ModRefSummary_->createModRefSet();

  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [&](const rvsdg::StructuralNode & structuralNode)
        {
          const auto nodeModRefSet = AnnotateStructuralNode(structuralNode, lambda);
          AddModRefSimpleConstraint(nodeModRefSet, regionModRefSet);
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          if (const auto nodeModRefSet = AnnotateSimpleNode(simpleNode, lambda))
            AddModRefSimpleConstraint(*nodeModRefSet, regionModRefSet);
        });
  }

  // Check if this region has any non-reentrant allocas. If so, block them
  if (const auto it = Context_->NonReentrantAllocas.find(&region);
      it != Context_->NonReentrantAllocas.end() && ENABLE_NON_REENTRANT_ALLOCA_BLOCKLIST)
  {
    JLM_ASSERT(ModRefSummary_->getModRefSet(regionModRefSet).IsEmpty());
    AddModRefSetBlocklist(regionModRefSet, it->second);
  }

  return regionModRefSet;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateStructuralNode(
    const rvsdg::StructuralNode & structuralNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRefSet = ModRefSummary_->getOrCreateSetForNode(structuralNode);

  for (auto & subregion : structuralNode.Subregions())
  {
    const auto subregionModRefSef = AnnotateRegion(subregion, lambda);
    AddModRefSimpleConstraint(subregionModRefSef, nodeModRefSet);
  }

  return nodeModRefSet;
}

std::optional<ModRefSetIndex>
RegionAwareModRefSummarizer::AnnotateSimpleNode(
    const rvsdg::SimpleNode & simpleNode,
    const rvsdg::LambdaNode & lambda)
{
  return MatchTypeWithDefault(
      simpleNode.GetOperation(),
      [&](const LoadOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateLoad(simpleNode, lambda);
      },
      [&](const StoreOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateStore(simpleNode, lambda);
      },
      [&](const AllocaOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateAlloca(simpleNode);
      },
      [&](const MallocOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateMalloc(simpleNode);
      },
      [&](const FreeOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateFree(simpleNode, lambda);
      },
      [&](const MemCpyOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateMemcpy(simpleNode, lambda);
      },
      [&](const MemSetOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateMemset(simpleNode, lambda);
      },
      [&](const CallOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateCall(simpleNode, lambda);
      },
      [&](const MemoryStateOperation &) -> std::optional<ModRefSetIndex>
      {
        // MemoryStateOperations are only used to route memory states, and can be ignored
        return std::nullopt;
      },
      [&]() -> std::optional<ModRefSetIndex>
      {
        // Any remaining type of node should not involve any memory states
        JLM_ASSERT(!hasMemoryState(simpleNode));
        return std::nullopt;
      });
}

void
RegionAwareModRefSummarizer::AddPointerOriginTargets(
    ModRefSetIndex modRefSetIndex,
    const rvsdg::Output & origin,
    std::optional<size_t> minTargetSize,
    const rvsdg::LambdaNode & lambda)
{
  const auto & pointsToGraph = Context_->pointsToGraph;
  const auto & allocasDead = Context_->AllocasDeadInScc[Context_->FunctionToSccIndex[&lambda]];

  // TODO: Re-use ModRefSets for all uses of the registerNode in this function
  const auto registerPtgNode = pointsToGraph.getNodeForRegister(origin);

  const auto tryAddToModRefSet = [&](PointsToGraph::NodeIndex targetPtgNode)
  {
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && pointsToGraph.isNodeConstant(targetPtgNode))
      return;
    if (ENABLE_OPERATION_SIZE_BLOCKING && minTargetSize)
    {
      const auto targetSize = pointsToGraph.tryGetNodeSize(targetPtgNode);
      if (targetSize && *targetSize < minTargetSize)
        return;
    }
    if (ENABLE_DEAD_ALLOCA_BLOCKLIST && allocasDead.Contains(targetPtgNode))
      return;
    ModRefSummary_->AddToModRefSet(modRefSetIndex, targetPtgNode);
  };

  // If the pointer is targeting everything external, add it all to the Mod/Ref set
  if (pointsToGraph.isTargetingAllExternallyAvailable(registerPtgNode))
  {
    for (const auto targetPtgNode : pointsToGraph.getExternallyAvailableNodes())
    {
      tryAddToModRefSet(targetPtgNode);
    }
  }

  for (const auto targetPtgNode : pointsToGraph.getExplicitTargets(registerPtgNode).Items())
  {
    tryAddToModRefSet(targetPtgNode);
  }
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateLoad(
    const rvsdg::SimpleNode & loadNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(loadNode);
  const auto origin = LoadOperation::AddressInput(loadNode).origin();
  const auto loadOperation = util::assertedCast<const LoadOperation>(&loadNode.GetOperation());
  const auto loadSize = GetTypeStoreSize(*loadOperation->GetLoadedType());

  AddPointerOriginTargets(nodeModRef, *origin, loadSize, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateStore(
    const rvsdg::SimpleNode & storeNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(storeNode);
  const auto origin = StoreOperation::AddressInput(storeNode).origin();
  const auto storeOperation = util::assertedCast<const StoreOperation>(&storeNode.GetOperation());
  const auto storeSize = GetTypeStoreSize(storeOperation->GetStoredType());

  AddPointerOriginTargets(nodeModRef, *origin, storeSize, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateAlloca(const rvsdg::SimpleNode & allocaNode)
{
  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(allocaNode);
  const auto allocaMemoryNode = Context_->pointsToGraph.getNodeForAlloca(allocaNode);
  ModRefSummary_->AddToModRefSet(nodeModRef, allocaMemoryNode);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMalloc(const rvsdg::SimpleNode & mallocNode)
{
  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(mallocNode);
  const auto mallocMemoryNode = Context_->pointsToGraph.getNodeForMalloc(mallocNode);
  ModRefSummary_->AddToModRefSet(nodeModRef, mallocMemoryNode);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateFree(
    const rvsdg::SimpleNode & freeNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<FreeOperation>(freeNode.GetOperation()));

  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(freeNode);
  const auto origin = FreeOperation::addressInput(freeNode).origin();

  // TODO: Only free MallocMemoryNodes
  AddPointerOriginTargets(nodeModRef, *origin, std::nullopt, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMemcpy(
    const rvsdg::SimpleNode & memcpyNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<MemCpyOperation>(memcpyNode.GetOperation()));

  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(memcpyNode);
  const auto dstOrigin = MemCpyOperation::destinationInput(memcpyNode).origin();
  const auto srcOrigin = MemCpyOperation::sourceInput(memcpyNode).origin();
  const auto countOrigin = MemCpyOperation::countInput(memcpyNode).origin();
  const auto count = tryGetConstantSignedInteger(*countOrigin);
  AddPointerOriginTargets(nodeModRef, *dstOrigin, count, lambda);
  AddPointerOriginTargets(nodeModRef, *srcOrigin, count, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMemset(
    const rvsdg::SimpleNode & memsetNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<MemSetOperation>(memsetNode.GetOperation()));

  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(memsetNode);
  const auto dstOrigin = MemSetOperation::destinationInput(memsetNode).origin();
  const auto lengthOrigin = MemSetOperation::lengthInput(memsetNode).origin();
  const auto numBytes = tryGetConstantSignedInteger(*lengthOrigin);
  AddPointerOriginTargets(nodeModRef, *dstOrigin, numBytes, lambda);

  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateCall(
    const rvsdg::SimpleNode & callNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<CallOperation>(callNode.GetOperation()));

  const auto & pointsToGraph = Context_->pointsToGraph;

  // This ModRefSet represents everything the call may affect
  const auto callModRef = ModRefSummary_->getOrCreateSetForNode(callNode);

  // Go over all possible targets of the call and add them to the call summary
  const auto targetPtr = callNode.input(0)->origin();
  const auto targetPtgNode = Context_->pointsToGraph.getNodeForRegister(*targetPtr);

  // Go through all locations the called function pointer may target
  for (const auto calleePtgNode : pointsToGraph.getExplicitTargets(targetPtgNode).Items())
  {
    const auto kind = pointsToGraph.getNodeKind(calleePtgNode);
    if (kind == PointsToGraph::NodeKind::LambdaNode)
    {
      const auto & calleeLambda = pointsToGraph.getLambdaForNode(calleePtgNode);
      const auto targetModRefSet = ModRefSummary_->GetOrCreateSetForNode(calleeLambda);
      AddModRefSimpleConstraint(targetModRefSet, callModRef);
    }
    else if (kind == PointsToGraph::NodeKind::ImportNode)
    {
      AddModRefSimpleConstraint(ModRefSummary_->getExternModRefSet(), callModRef);
    }
  }
  if (pointsToGraph.isTargetingAllExternallyAvailable(targetPtgNode))
  {
    AddModRefSimpleConstraint(ModRefSummary_->getExternModRefSet(), callModRef);
  }

  // Allocas that are live within the call, might no longer be live from the call site
  if (ENABLE_DEAD_ALLOCA_BLOCKLIST)
  {
    JLM_ASSERT(ModRefSummary_->getModRefSet(callModRef).IsEmpty());
    AddModRefSetBlocklist(
        callModRef,
        Context_->AllocasDeadInScc[Context_->FunctionToSccIndex[&lambda]]);
  }

  return callModRef;
}

void
RegionAwareModRefSummarizer::SolveModRefSetConstraintGraph()
{
  Context_->ModRefSetSimpleConstraints.resize(ModRefSummary_->NumModRefSets());
  util::TwoPhaseLrfWorklist<ModRefSetIndex> worklist;

  // Start by pushing everything to the worklist
  for (ModRefSetIndex i = 0; i < ModRefSummary_->NumModRefSets(); i++)
    worklist.PushWorkItem(i);

  while (worklist.HasMoreWorkItems())
  {
    const auto workItem = worklist.PopWorkItem();

    // Handle all simple constraints workItem -> target
    for (auto target : Context_->ModRefSetSimpleConstraints[workItem].Items())
    {
      bool changed = false;
      if (auto blocklist = Context_->ModRefSetBlocklists.find(target);
          blocklist != Context_->ModRefSetBlocklists.end())
      {
        // The target has a blocklist, avoid propagating blocked MemoryNodes
        for (auto memoryNode : ModRefSummary_->GetModRefSet(workItem).Items())
        {
          if (blocklist->second->Contains(memoryNode))
            continue;
          changed |= ModRefSummary_->AddToModRefSet(target, memoryNode);
        }
      }
      else
      {
        // The target has no blocklist, propagate everything
        changed |= ModRefSummary_->PropagateModRefSet(workItem, target);
      }

      if (changed)
        worklist.PushWorkItem(target);
    }
  }

  JLM_ASSERT(VerifyBlocklists());
}

bool
RegionAwareModRefSummarizer::VerifyBlocklists() const
{
  // For all ModRefSets where a blocklist has been defined,
  // check that none of its MemoryNodes are on the blocklist
  for (auto [index, blocklist] : Context_->ModRefSetBlocklists)
  {
    for (auto memoryNode : ModRefSummary_->GetModRefSet(index).Items())
    {
      if (blocklist->Contains(memoryNode))
        return false;
    }
  }
  return true;
}

void
RegionAwareModRefSummarizer::doExternalCompression()
{
  for (auto & functions : Context_->SccFunctions)
  {
    for (auto function : functions.Items())
    {
      compressExternalInFunction(*function);
    }
  }
}

void
RegionAwareModRefSummarizer::compressExternalInFunction(const rvsdg::LambdaNode & lambda)
{
  // The set of ModRefSets that belong to this function
  // TODO: They could be placed in a map when created to avoid needing to find them again
  util::HashSet<ModRefSetIndex> modRefSets;
  findAllModRefSets(lambda, modRefSets);

  // The node representing all memory locations that are only known in external modules
  const auto externalMemoryNode = aa::PointsToGraph::externalMemoryNode;

  // Memory nodes that appear in modRefSets without the external memory node
  util::HashSet<PointsToGraph::NodeIndex> memoryNodesToKeep;
  for (auto modRefSetIndex : modRefSets.Items())
  {
    auto & modRefSet = ModRefSummary_->GetModRefSet(modRefSetIndex);
    if (modRefSet.Contains(externalMemoryNode))
      continue;

    for (auto memoryNode : modRefSet.Items())
      memoryNodesToKeep.insert(memoryNode);
  }

  // All memory nodes that are compressed are instead represented by external, which we keep
  memoryNodesToKeep.insert(externalMemoryNode);

  // Now go over again, removing all memory nodes that are not on the keep list
  for (auto modRefSetIndex : modRefSets.Items())
  {
    auto & modRefSet = ModRefSummary_->GetModRefSet(modRefSetIndex);
    if (!modRefSet.Contains(externalMemoryNode))
    {
      Context_->numLocalModRefSets++;
      Context_->numLocalModRefKept += modRefSet.Size();
      continue;
    }

    Context_->numExtModRefSets++;
    size_t removed = modRefSet.RemoveWhere(
        [&](PointsToGraph::NodeIndex node)
        {
          return !memoryNodesToKeep.Contains(node);
        });
    Context_->numExtModRefCompressed += removed;
    Context_->numExtModRefKept += modRefSet.Size();
  }
}

void
RegionAwareModRefSummarizer::findAllModRefSets(
    const rvsdg::Node & node,
    util::HashSet<ModRefSetIndex> & modRefSets)
{
  if (ModRefSummary_->hasSetForNode(node))
  {
    modRefSets.insert(ModRefSummary_->getSetForNode(node));
  }

  // If the node is a structural node, recurse into its subregion(s)
  rvsdg::MatchType(
      node,
      [&](const rvsdg::StructuralNode & structuralNode)
      {
        for (auto & subregion : structuralNode.Subregions())
        {
          for (auto & node : subregion.Nodes())
          {
            findAllModRefSets(node, modRefSets);
          }
        }
      });
}

std::string
RegionAwareModRefSummarizer::CallGraphSCCsToString(const RegionAwareModRefSummarizer & summarizer)
{
  std::ostringstream ss;
  for (size_t i = 0; i < summarizer.Context_->SccFunctions.size(); i++)
  {
    if (i != 0)
      ss << " <- ";
    ss << "[" << std::endl;
    if (i == summarizer.Context_->ExternalNodeSccIndex)
    {
      ss << "  " << "<external>" << std::endl;
    }
    for (auto function : summarizer.Context_->SccFunctions[i].Items())
    {
      ss << "  " << function->DebugString() << std::endl;
    }
    ss << "]";
  }
  return ss.str();
}

std::string
RegionAwareModRefSummarizer::ToRegionTree(
    const rvsdg::Graph & rvsdg,
    const RegionAwareModRefSummary & modRefSummary)
{
  const auto & pointsToGraph = modRefSummary.GetPointsToGraph();

  std::ostringstream ss;

  auto toString = [&](const util::HashSet<PointsToGraph::NodeIndex> & memoryNodes)
  {
    ss << "MemoryNodes: {";
    for (auto & memoryNode : memoryNodes.Items())
    {
      ss << pointsToGraph.getNodeDebugString(memoryNode);
      ss << ", ";
    }
    ss << "}" << std::endl;
  };

  auto indent = [&](size_t depth, char c = '-')
  {
    for (size_t i = 0; i < depth; i++)
      ss << c;
  };

  std::function<void(const rvsdg::Node &, size_t)> toRegionTree =
      [&](const rvsdg::Node & node, size_t depth)
  {
    if (!modRefSummary.HasSetForNode(node))
      return;

    auto modRefIndex = modRefSummary.GetSetForNode(node);
    auto & memoryNodes = modRefSummary.GetModRefSet(modRefIndex);
    ss << " ";
    toString(memoryNodes);

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
