/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

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
#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/TarjanScc.hpp>
#include <jlm/util/Worklist.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_map>

namespace jlm::llvm::aa
{
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

  static constexpr auto NumModRefSetsMaterializedLabel_ = "#ModRefSetsMaterialized";
  static constexpr auto ModRefSetSizeBeforeMaterializationLabel_ =
      "ModRefSetSizeBeforeMaterialization";
  static constexpr auto ModRefSetSizeAfterFilteringLabel_ = "ModRefSetSizeAfterFiltering";
  static constexpr auto NumModRefSetsWithEffectOnExternalLabel_ = "#ModRefSetsWithEffectOnExternal";
  static constexpr auto NumModRefSetsCallingExternalFunctionLabel_ =
      "#ModRefSetsCallingExternalFunction";
  static constexpr auto ModRefSetSizeAfterMaterializationLabel_ =
      "ModRefSetSizeAfterMaterialization";

  static constexpr auto CallGraphTimer_ = "CallGraphTimer";
  static constexpr auto AllocasDeadInSccsTimer_ = "AllocasDeadInSccsTimer";
  static constexpr auto SimpleAllocasSetTimer_ = "SimpleAllocasSetTimer";
  static constexpr auto NonReentrantAllocaSetsTimer_ = "NonReentrantAllocaSetsTimer";
  static constexpr auto AnnotationTimer_ = "AnnotationTimer";
  static constexpr auto SolvingTimer_ = "SolvingTimer";
  static constexpr auto ModRefSetMaterializationTimer_ = "ModRefSetMaterializationTimer";

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
  startModRefSetMaterializationStatistics()
  {
    AddTimer(ModRefSetMaterializationTimer_).start();
  }

  void
  stopModRefSetMaterializationStatistics(
      size_t numModRefSetsMaterialized,
      size_t modRefSetSizeBeforeMaterialization,
      size_t modRefSetSizeAfterFiltering,
      size_t numModRefSetsWithEffectOnExternal,
      size_t numModRefSetsCallingExternalFunction,
      size_t modRefSetSizeAfterMaterialization)
  {
    GetTimer(ModRefSetMaterializationTimer_).stop();
    AddMeasurement(NumModRefSetsMaterializedLabel_, numModRefSetsMaterialized);
    AddMeasurement(ModRefSetSizeBeforeMaterializationLabel_, modRefSetSizeBeforeMaterialization);
    AddMeasurement(ModRefSetSizeAfterFilteringLabel_, modRefSetSizeAfterFiltering);
    AddMeasurement(NumModRefSetsWithEffectOnExternalLabel_, numModRefSetsWithEffectOnExternal);
    AddMeasurement(
        NumModRefSetsCallingExternalFunctionLabel_,
        numModRefSetsCallingExternalFunction);
    AddMeasurement(ModRefSetSizeAfterMaterializationLabel_, modRefSetSizeAfterMaterialization);
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
   * The set can be flagged as possibly referencing all externally available memory locations
   * with a byte size >= some minimum. (Also includes all memory nodes of unknown size.)
   * If the set is flagged as possibly calling external functions, its size will always be 0,
   * indicating that all externally available memory is possibly referenced by this set.
   *
   * @return the minimum size where all externally available memory is referenced by this set.
   *         If the set is not flagged, nullopt is returned.
   */
  [[nodiscard]] std::optional<size_t>
  getRefExternalMinSize() const
  {
    if (refExternalOfSize_ == NoneSize)
      return std::nullopt;
    return refExternalOfSize_;
  }

  /**
   * The set can be flagged as possibly modifying all externally available memory locations
   * with a byte size >= some minimum. (Also includes all memory nodes of unknown size.)
   * If the set is flagged as possibly calling external functions, the size will be 0,
   * indicating that all externally available memory is possibly modified by this set.
   *
   * @return the minimum size where all externally available memory is modified by this set.
   *         If the set is not flagged, nullopt is returned.
   */
  [[nodiscard]] std::optional<size_t>
  getModExternalMinSize() const
  {
    if (modExternalOfSize_ == NoneSize)
      return std::nullopt;
    return modExternalOfSize_;
  }

  /**
   * Marks the ModRefSet as referencing all externally available memory of size >= minSize.
   * @param minSize the minimum byte size of the memory locations. 0 means all sizes.
   * @return true if the ModRefSet was modified by this operation, otherwise false
   */
  bool
  markAsReferencingExternal(size_t minSize)
  {
    // Make sure we do not overflow or collide with the sentinel NoneSize
    minSize = std::min<size_t>(minSize, NoneSize - 1);

    if (minSize < refExternalOfSize_)
    {
      refExternalOfSize_ = minSize;
      return true;
    }
    return false;
  }

  /**
   * Marks the set as possibly modifying all externally available memory of size >= minSize.
   * @param minSize the minimum byte size of the memory locations. 0 means all sizes.
   * @return true if the ModRefSet was modified by this operation, otherwise false
   */
  bool
  markAsModifyingExternal(size_t minSize)
  {
    // Make sure we do not overflow or collide with the sentinel NoneSize
    minSize = std::min<size_t>(minSize, NoneSize - 1);

    if (minSize < modExternalOfSize_)
    {
      modExternalOfSize_ = minSize;
      return true;
    }
    return false;
  }

  /**
   * Helper function for finding the implicitly flagged \ref ModRefEffect on a hypothetical
   * externally available memory node, optionally of a specific size.
   * @param size the size of the hypothetical memory node, or nullopt if unknown
   * @return the implicitly encoded ModRefEffect on the hypothetical memory node
   */
  [[nodiscard]] ModRefEffect
  getImplicitModRefEffectForExternal(std::optional<size_t> size) const noexcept
  {
    ModRefEffect result = ModRefEffect::NoEffect;
    if (refExternalOfSize_ != NoneSize)
    {
      if (size.value_or(NoneSize) >= refExternalOfSize_)
        result |= ModRefEffect::RefOnly;
    }
    if (modExternalOfSize_ != NoneSize)
    {
      if (size.value_or(NoneSize) >= modExternalOfSize_)
        result |= ModRefEffect::ModOnly;
    }
    return result;
  }

  /**
   * @return true if the \ref ModRefSet represents possible calls to externally defined functions.
   */
  [[nodiscard]] bool
  mayCallExternalFunction() const
  {
    return callsExternalFunction_;
  }

  /**
   * Marks the \ref ModRefSet as possibly containing calls to externally defined functions.
   * @return true if the \ref ModRefSet was modified by this operation, otherwise false
   */
  bool
  markAsCallingExternalFunction()
  {
    if (callsExternalFunction_)
      return false;

    callsExternalFunction_ = true;

    // calls to external functions may reference and modify any externally available memory
    refExternalOfSize_ = 0;
    modExternalOfSize_ = 0;
    return true;
  }

  /**
   * Adds the given \p memoryNode as an explicit member of the \ref ModRefSet.
   * This function does not attempt to skip memory nodes that are already implicitly referenced.
   *
   * @param memoryNode the index of the memory node in the points-to graph.
   * @param modRefEffect the effect(s) that may be performed on the memory node
   * @return true if the ModRefSet was modified by this operation, otherwise false
   */
  bool
  addExplicitMemoryNode(PointsToGraph::NodeIndex memoryNode, ModRefEffect modRefEffect)
  {
    JLM_ASSERT(modRefEffect != ModRefEffect::NoEffect);

    const auto [it, inserted] = modRefNodes_.insert({ memoryNode, modRefEffect });
    if (inserted)
      return true;

    // The memory node was already present, but we may add more effects
    auto oldEffects = it->second;
    it->second = oldEffects | modRefEffect;
    return it->second != oldEffects;
  }

  /**
   * Adds the given memoryNode to the set, unless the memory node is already represented.
   * Uses the external availability and byte size of the memory node to determine this.
   *
   * @param memoryNode the index of the memory node in the points-to graph.
   * @param isExternallyAvailable a boolean indicating if the memory node is externally available.
   * @param memoryNodeSize the byte size of the memory node, if known. Otherwise nullopt.
   * @param modRefEffect the effect that may be performed on the added memory node.
   * @return true if the ModRefSet was modified by this operation, otherwise false
   */
  bool
  addMemoryNode(
      PointsToGraph::NodeIndex memoryNode,
      bool isExternallyAvailable,
      std::optional<size_t> memoryNodeSize,
      ModRefEffect modRefEffect)
  {
    JLM_ASSERT(modRefEffect != ModRefEffect::NoEffect);

    // If the effects on the node are already encoded implicitly, skip adding them explicitly
    if (isExternallyAvailable)
    {
      const auto implicitEffect = getImplicitModRefEffectForExternal(memoryNodeSize);
      if (isEffectSubset(modRefEffect, implicitEffect))
        return false;
    }

    return addExplicitMemoryNode(memoryNode, modRefEffect);
  }

  /**
   * Removes all explicit memory nodes that are not included in the given \p filter.
   */
  void
  keepSubsetOfExplicitMemoryNodes(const util::HashSet<PointsToGraph::NodeIndex> & filter)
  {
    auto it = modRefNodes_.begin();
    while (it != modRefNodes_.end())
    {
      if (filter.Contains(it->first))
        ++it;
      else
        it = modRefNodes_.erase(it);
    }
  }

  /**
   * Propagates all flags from the given set \p other to this set.
   * @return true if any flags in this ModRefSet changed, false otherwise
   */
  bool
  propagateFlags(const RegionAwareModRefSet & other)
  {
    // Check the most restrictive flag first
    if (other.callsExternalFunction_)
    {
      if (callsExternalFunction_)
        return false;

      callsExternalFunction_ = true;
      JLM_ASSERT(other.refExternalOfSize_ == 0);
      JLM_ASSERT(other.modExternalOfSize_ == 0);
      refExternalOfSize_ = 0;
      modExternalOfSize_ = 0;
      return true;
    }

    if (other.modExternalOfSize_ < modExternalOfSize_)
    {
      modExternalOfSize_ = other.modExternalOfSize_;
      refExternalOfSize_ = std::min(refExternalOfSize_, other.refExternalOfSize_);
      return true;
    }

    if (other.refExternalOfSize_ < refExternalOfSize_)
    {
      refExternalOfSize_ = other.refExternalOfSize_;
      return true;
    }

    return false;
  }

private:
  // If not equal to NoneSize, the ModRefSet potentially reads all externally available memory
  // of size >= the given number of bytes.
  uint32_t refExternalOfSize_ = NoneSize;
  // If not equal to NoneSize, the ModRefSet potentially modifies all externally available memory
  // of size >= the given number of bytes.
  uint32_t modExternalOfSize_ = NoneSize;
  // If true, the ModRefSet represents possibly calling externally defined functions.
  // In this case, \ref refExternalOfSize and \ref modExternalOfSize must both be 0
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
    // Create the ModRefSet representing eveything that can be referenced and modified,
    // directly or indirectly, from external functions.
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

  bool
  markSetAsReferencingExternal(ModRefSetIndex index, size_t minSize)
  {
    return getModRefSet(index).markAsReferencingExternal(minSize);
  }

  bool
  markSetAsModifyingExternal(ModRefSetIndex index, size_t minSize)
  {
    return getModRefSet(index).markAsModifyingExternal(minSize);
  }

  bool
  markSetAsCallingExternalFunction(ModRefSetIndex index)
  {
    return getModRefSet(index).markAsCallingExternalFunction();
  }

  /**
   * Adds the given \p ptgNode the \ref ModRefSet with the given \p index.
   * Performs checks to avoid adding doubled-up memory nodes.
   *
   * @param index the index of the \ref ModRefSet being added to
   * @param ptgNode the index of the memory node in the points-to graph
   * @param modRefEffect the effect the \ref ModRefSet has on the memory node
   * @return true if the \ref ModRefSet grew, false otherwise.
   */
  bool
  addMemoryNodeToSet(
      ModRefSetIndex index,
      PointsToGraph::NodeIndex ptgNode,
      ModRefEffect modRefEffect)
  {
    const bool isExternallyAvailable = pointsToGraph_.isExternallyAvailable(ptgNode);
    const std::optional<size_t> memoryNodeSize = pointsToGraph_.tryGetNodeSize(ptgNode);
    return getModRefSet(index)
        .addMemoryNode(ptgNode, isExternallyAvailable, memoryNodeSize, modRefEffect);
  }

  /**
   * Adds the given memory node to a \ref ModRefSet with the given effect.
   * Does not perform any checks against doubled-up memory nodes.
   *
   * @param index the index of the \ref ModRefSet being added to
   * @param ptgNode the index of the memory node in the points-to graph
   * @param modRefEffect the effect the \ref ModRefSet has on the memory node
   * @see addMemoryNodeToSet
   */
  bool
  addExplicitMemoryNodeToSet(
      ModRefSetIndex index,
      PointsToGraph::NodeIndex ptgNode,
      ModRefEffect modRefEffect)
  {
    return getModRefSet(index).addExplicitMemoryNode(ptgNode, modRefEffect);
  }

  /**
   * The ModRefSummary has one set representing external functions, containing all
   * memory nodes that can be referenced or modified by external functions.
   * This set also includes memory operations performed in functions that can be called from
   * external functions, so will in practice contain most memory operations in the module.
   * The exceptions are allocas that are provably not involved in recursion,
   * since any such alloca that is affected by a call to an external function will be dead
   * by the time the call returns.
   *
   * @return the index of the \ref ModRefSet representing external functions
   */
  [[nodiscard]] ModRefSetIndex
  getExternModRefSet() const noexcept
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
   * Get the \ref ModRefSet associated with the given \p node, or creates one if none exists.
   * @param node the RVSDG node to be represented by a ModRefSet
   * @param lambdaNode the function the RVSDG node belongs to.
   * @return the index of the exisiting or created \ref ModRefSet representing the node
   */
  [[nodiscard]] ModRefSetIndex
  getOrCreateSetForNode(const rvsdg::Node & node, const rvsdg::LambdaNode & lambdaNode)
  {
    auto [it, inserted] = nodeMap_.insert({ &node, 0 });
    if (inserted)
    {
      const auto created = createModRefSet();
      it->second = created;
      modRefSetsInFunction_[&lambdaNode].push_back(created);
      return created;
    }

    return it->second;
  }

  [[nodiscard]] const std::vector<ModRefSetIndex> &
  getAllSetsInFunction(const rvsdg::LambdaNode & lambdaNode)
  {
    return modRefSetsInFunction_[&lambdaNode];
  }

  [[nodiscard]] std::string
  getModRefSetDebugString(ModRefSetIndex index) const
  {
    const auto & modRefSet = getModRefSet(index);
    std::stringstream ss;
    ss << "{MRS#" << index << "; ";
    const auto mayRefMinSize = modRefSet.getRefExternalMinSize();
    if (mayRefMinSize.has_value())
      ss << "RefExt>=" << *mayRefMinSize << " bytes; ";
    const auto mayModMinSize = modRefSet.getModExternalMinSize();
    if (mayModMinSize.has_value())
      ss << "ModExt>=" << *mayModMinSize << " bytes; ";
    if (modRefSet.mayCallExternalFunction())
      ss << "MayCallExt; ";

    bool first = true;
    for (auto [memoryNode, modRefEffect] : modRefSet.getModRefNodes())
    {
      if (first)
        first = false;
      else
        ss << ", ";

      ss << pointsToGraph_.getNodeDebugString(memoryNode);
      switch (modRefEffect)
      {
      case jlm::llvm::aa::ModRefEffect::NoEffect:
        JLM_UNREACHABLE("Memory nodes should never be added with NoEffect");
      case jlm::llvm::aa::ModRefEffect::RefOnly:
        ss << "[R]";
        break;
      case jlm::llvm::aa::ModRefEffect::ModOnly:
        ss << "[M]";
        break;
      case jlm::llvm::aa::ModRefEffect::ModRef:
        ss << "[MR]";
        break;
      }
    }
    ss << "}";
    return ss.str();
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
  [[nodiscard]] ModRefSetIndex
  createModRefSet()
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
   * The \ref ModRefSet representing every effect external function calls may have on memory nodes.
   * These effects include the possibility of external functions calling into local functions.
   * @see getExternModRefSet()
   */
  ModRefSetIndex externModRefSet_;

  /**
   * Lists \ref ModRefSet%s grouped by the function of node they belong to.
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
   * The set of all Simple Allocas in the module.
   *
   * Assigned in \ref CreateSimpleAllocaSet(). Remains constant after.
   */
  util::HashSet<PointsToGraph::NodeIndex> SimpleAllocas;

  /**
   * For structural nodes whose subregion(s) contain alloca nodes that are non-reentrant,
   * the memory nodes respresenting those alloca nodes are added to this map.
   *
   * Assigned in \ref CreateNonReentrantAllocaSets(). Remains constant after.
   */
  std::unordered_map<const rvsdg::Node *, util::HashSet<PointsToGraph::NodeIndex>>
      NonReentrantAllocas;

  /**
   * Simple edges in the \ref ModRefSet constraint graph.
   * A simple edge a -> b indicates that the \ref ModRefSet b should contain everything in a.
   * ModRefSetSimpleEdges[a] contains b, as well as any other simple edge successors.
   */
  std::vector<util::HashSet<ModRefSetIndex>> ModRefSetSimpleConstraints;

  /**
   * Blocklists in the \ref ModRefSet constraint graph.
   * During solving, a memory node X that is about to be propagated to a \ref ModRefSet A
   * will be skipped if X is in the blocklist associated with A.
   * The pointer to the blocklist must remain valid until solving is finished.
   */
  std::unordered_map<ModRefSetIndex, const util::HashSet<PointsToGraph::NodeIndex> *>
      ModRefSetBlocklists;

  /**
   * The number of \ref ModRefSet%s that go through materialization.
   * Should be all sets except for the one representing external function calls.
   */
  size_t numModRefSetsMaterialized = 0;
  /**
   * The total number of explicit memory nodes in all \ref ModRefSet%s before materialization.
   */
  size_t modRefSetSizeBeforeMaterialization = 0;
  /**
   * The total number of explicit memory nodes in all \ref ModRefSet%s after filtering
   * away memory nodes that can be compressed away.
   */
  size_t modRefSetSizeAfterFiltering = 0;
  /**
   * The number of \ref ModRefSet%s that are flagged as having an
   * effect on externally available memory.
   */
  size_t numModRefSetsWithEffectOnExternal = 0;
  /**
   * The number of \ref ModRefSet%s that are flagged as possibly calling an external function.
   */
  size_t numModRefSetsCallingExternalFunction = 0;
  /**
   * The total number of explicit memory nodes in all \ref ModRefSet%s after materialization.
   * For sets that are not flagged as effecting externally available memory,
   * their size after materialization will be equal to their size after filtering.
   */
  size_t modRefSetSizeAfterMaterialization = 0;
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

  statistics->StartCreateSimpleAllocasSetStatistics();
  Context_->SimpleAllocas = CreateSimpleAllocaSet(pointsToGraph);
  statistics->StopCreateSimpleAllocasSetStatistics(Context_->SimpleAllocas.Size());

  statistics->StartCreateNonReentrantAllocaSetsStatistics();
  auto numNonReentrantAllocas = CreateNonReentrantAllocaSets();
  statistics->StopCreateNonReentrantAllocaSetsStatistics(numNonReentrantAllocas);

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

  // Print debug output
  // std::cerr << PointsToGraph::dumpDot(pointsToGraph) << std::endl;
  // std::cerr << "numSimpleAllocas: " << Context_->SimpleAllocas.Size() << std::endl;
  // std::cerr << "numNonReentrantAllocas: " << numNonReentrantAllocas << std::endl;
  // std::cerr << "Call Graph SCCs:" << std::endl << CallGraphSCCsToString(*this) << std::endl;
  // std::cerr << "After solving, before materialization: " << std::endl;
  // std::cerr << ToRegionTree(rvsdgModule.Rvsdg(), *ModRefSummary_) << std::endl;

  statistics->startModRefSetMaterializationStatistics();
  materializeSets();
  statistics->stopModRefSetMaterializationStatistics(
      Context_->numModRefSetsMaterialized,
      Context_->modRefSetSizeBeforeMaterialization,
      Context_->modRefSetSizeAfterFiltering,
      Context_->numModRefSetsWithEffectOnExternal,
      Context_->numModRefSetsCallingExternalFunction,
      Context_->modRefSetSizeAfterMaterialization);

  // More debug output
  // std::cerr << "After materialization: " << std::endl;
  // std::cerr << ToRegionTree(rvsdgModule.Rvsdg(), *ModRefSummary_) << std::endl;

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
RegionAwareModRefSummarizer::getReachableSimpleAllocas(std::queue<PointsToGraph::NodeIndex> & nodes)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  util::HashSet<PointsToGraph::NodeIndex> reachableSimpleAllocas;
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

util::HashSet<PointsToGraph::NodeIndex>
RegionAwareModRefSummarizer::getSimpleAllocasReachableFromRegionArguments(
    const rvsdg::Region & region)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // Start by finding initial register nodes
  std::queue<PointsToGraph::NodeIndex> nodes;
  for (auto argument : region.Arguments())
  {
    if (!IsPointerCompatible(*argument))
      continue;
    const auto ptgNode = pointsToGraph.getNodeForRegister(*argument);
    nodes.push(ptgNode);
  }

  return getReachableSimpleAllocas(nodes);
}

util::HashSet<PointsToGraph::NodeIndex>
RegionAwareModRefSummarizer::getSimpleAllocasReachableFromCallArguments(
    const rvsdg::SimpleNode & call)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // Use a queue and a set to traverse the PointsToGraph
  std::queue<PointsToGraph::NodeIndex> nodes;
  auto numArguments = CallOperation::NumArguments(call);
  for (size_t i = 0; i < numArguments; i++)
  {
    const auto & argument = *CallOperation::Argument(call, i)->origin();

    if (!IsPointerCompatible(argument))
      continue;
    const auto ptgNode = pointsToGraph.getNodeForRegister(argument);
    nodes.push(ptgNode);
  }

  return getReachableSimpleAllocas(nodes);
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
    return reachableSimpleAllocas[&region] = getSimpleAllocasReachableFromRegionArguments(region);
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
    if (!isNonReentrant(simpleAllocaPtgNode))
      continue;

    const auto & region = *pointsToGraph.getAllocaForNode(simpleAllocaPtgNode).region();
    const auto structuralNode = region.node();
    // Creates a set for the structural node if it does not already have one, and add the alloca
    Context_->NonReentrantAllocas[structuralNode].insert(simpleAllocaPtgNode);
    numNonReentrantAllocas++;
  }

  return numNonReentrantAllocas;
}

void
RegionAwareModRefSummarizer::AddModRefSimpleConstraint(ModRefSetIndex from, ModRefSetIndex to)
{
  // We should never add outgoing edges from the set representing all external functions
  JLM_ASSERT(from != ModRefSummary_->getExternModRefSet());
  // Ensure the constraint vector is large enough
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
  const auto modRefSet = AnnotateStructuralNode(lambda, lambda);

  if (Context_->FunctionsCallingSetjmp.Contains(&lambda))
  {
    // If this function can be jumped into, store operations on memory in its Mod/Ref set must be
    // sequentialized with calls to external functions, in case the trigger jumps
    // TODO: This edge could in theory only propagate Mod info, and turn it into Ref info,
    // since calls to longjmp only need to be sequentialized with stores
    AddModRefSimpleConstraint(modRefSet, ModRefSummary_->getExternModRefSet());
  }

  // If the function is externally available, it can be called by external functions,
  // so add a simple edge to the ModRefSet representing all external functions.
  const auto lambdaPtgNode = Context_->pointsToGraph.getNodeForLambda(lambda);
  if (Context_->pointsToGraph.isExternallyAvailable(lambdaPtgNode))
  {
    AddModRefSimpleConstraint(modRefSet, ModRefSummary_->getExternModRefSet());
  }
}

void
RegionAwareModRefSummarizer::AnnotateRegion(
    const rvsdg::Region & region,
    ModRefSetIndex modRefSet,
    const rvsdg::LambdaNode & lambda)
{
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [&](const rvsdg::StructuralNode & structuralNode)
        {
          const auto nodeModRefSet = AnnotateStructuralNode(structuralNode, lambda);
          AddModRefSimpleConstraint(nodeModRefSet, modRefSet);
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          if (const auto nodeModRefSet = AnnotateSimpleNode(simpleNode, lambda))
            AddModRefSimpleConstraint(*nodeModRefSet, modRefSet);
        });
  }
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateStructuralNode(
    const rvsdg::StructuralNode & structuralNode,
    const rvsdg::LambdaNode & lambda)
{
  // The ModRefSet of a structural node is the same as that of its subregion(s)
  const auto modRefSet = ModRefSummary_->getOrCreateSetForNode(structuralNode, lambda);

  for (auto & subregion : structuralNode.Subregions())
  {
    AnnotateRegion(subregion, modRefSet, lambda);
  }

  // Check if this node has any non-reentrant allocas. If so, block them from leaving the node
  if (const auto it = Context_->NonReentrantAllocas.find(&structuralNode);
      it != Context_->NonReentrantAllocas.end() && ENABLE_NON_REENTRANT_ALLOCA_BLOCKLIST)
  {
    AddModRefSetBlocklist(modRefSet, it->second);
  }

  return modRefSet;
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
        return AnnotateAlloca(simpleNode, lambda);
      },
      [&](const MallocOperation &) -> std::optional<ModRefSetIndex>
      {
        return AnnotateMalloc(simpleNode, lambda);
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
RegionAwareModRefSummarizer::addPointerOriginTargets(
    ModRefSetIndex modRefSetIndex,
    const rvsdg::Output & origin,
    std::optional<size_t> minTargetSize,
    ModRefEffect modRefEffect)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  const auto registerPtgNode = pointsToGraph.getNodeForRegister(origin);

  const auto tryAddToModRefSet = [&](PointsToGraph::NodeIndex targetPtgNode)
  {
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && pointsToGraph.isNodeConstant(targetPtgNode))
      return;
    if (ENABLE_OPERATION_SIZE_BLOCKING && minTargetSize)
    {
      const auto targetSize = pointsToGraph.tryGetNodeSize(targetPtgNode);
      if (targetSize.has_value() && *targetSize < minTargetSize)
        return;
    }
    ModRefSummary_->addExplicitMemoryNodeToSet(modRefSetIndex, targetPtgNode, modRefEffect);
  };

  // If the pointer is targeting everything external, flag the ModRefSet
  if (pointsToGraph.isTargetingAllExternallyAvailable(registerPtgNode))
  {
    if (mayEffectReference(modRefEffect))
    {
      ModRefSummary_->markSetAsReferencingExternal(modRefSetIndex, minTargetSize.value_or(0));
    }
    if (mayEffectModify(modRefEffect))
    {
      ModRefSummary_->markSetAsModifyingExternal(modRefSetIndex, minTargetSize.value_or(0));
    }
  }

  for (const auto targetPtgNode : pointsToGraph.getExplicitTargets(registerPtgNode).Items())
  {
    // Assert that the PointsToGraph contains no doubled-up pointees
    JLM_ASSERT(
        !pointsToGraph.isTargetingAllExternallyAvailable(registerPtgNode)
        || !pointsToGraph.isExternallyAvailable(targetPtgNode));
    tryAddToModRefSet(targetPtgNode);
  }
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateLoad(
    const rvsdg::SimpleNode & loadNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(loadNode, lambda);
  const auto origin = LoadOperation::AddressInput(loadNode).origin();
  const auto loadOperation = util::assertedCast<const LoadOperation>(&loadNode.GetOperation());
  const auto loadSize = GetTypeStoreSize(*loadOperation->GetLoadedType());

  addPointerOriginTargets(nodeModRef, *origin, loadSize, ModRefEffect::RefOnly);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateStore(
    const rvsdg::SimpleNode & storeNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(storeNode, lambda);
  const auto origin = StoreOperation::AddressInput(storeNode).origin();
  const auto storeOperation = util::assertedCast<const StoreOperation>(&storeNode.GetOperation());
  const auto storeSize = GetTypeStoreSize(storeOperation->GetStoredType());

  addPointerOriginTargets(nodeModRef, *origin, storeSize, ModRefEffect::ModOnly);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateAlloca(
    const rvsdg::SimpleNode & allocaNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(allocaNode, lambda);
  const auto allocaMemoryNode = Context_->pointsToGraph.getNodeForAlloca(allocaNode);
  // The alloca itself is only considered to be a ref, since its value is indeterminite,
  // and any users of the alloca will depend on its address output
  ModRefSummary_->addExplicitMemoryNodeToSet(nodeModRef, allocaMemoryNode, ModRefEffect::RefOnly);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMalloc(
    const rvsdg::SimpleNode & mallocNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(mallocNode, lambda);
  const auto mallocMemoryNode = Context_->pointsToGraph.getNodeForMalloc(mallocNode);
  // The malloc itself is only considered to be a ref, since its value is indeterminite,
  // and any users of the malloc will depend on its address output
  ModRefSummary_->addExplicitMemoryNodeToSet(nodeModRef, mallocMemoryNode, ModRefEffect::RefOnly);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateFree(
    const rvsdg::SimpleNode & freeNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<FreeOperation>(freeNode.GetOperation()));

  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(freeNode, lambda);
  const auto origin = FreeOperation::addressInput(freeNode).origin();

  // TODO: Filter so we only free MallocMemoryNodes
  addPointerOriginTargets(nodeModRef, *origin, std::nullopt, ModRefEffect::ModOnly);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMemcpy(
    const rvsdg::SimpleNode & memcpyNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<MemCpyOperation>(memcpyNode.GetOperation()));

  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(memcpyNode, lambda);
  const auto dstOrigin = MemCpyOperation::destinationInput(memcpyNode).origin();
  const auto srcOrigin = MemCpyOperation::sourceInput(memcpyNode).origin();
  const auto countOrigin = MemCpyOperation::countInput(memcpyNode).origin();
  const auto count = tryGetConstantSignedInteger(*countOrigin);
  addPointerOriginTargets(nodeModRef, *dstOrigin, count, ModRefEffect::ModOnly);
  addPointerOriginTargets(nodeModRef, *srcOrigin, count, ModRefEffect::RefOnly);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMemset(
    const rvsdg::SimpleNode & memsetNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<MemSetOperation>(memsetNode.GetOperation()));

  const auto nodeModRef = ModRefSummary_->getOrCreateSetForNode(memsetNode, lambda);
  const auto dstOrigin = MemSetOperation::destinationInput(memsetNode).origin();
  const auto lengthOrigin = MemSetOperation::lengthInput(memsetNode).origin();
  const auto numBytes = tryGetConstantSignedInteger(*lengthOrigin);
  addPointerOriginTargets(nodeModRef, *dstOrigin, numBytes, ModRefEffect::ModOnly);

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
  const auto callModRef = ModRefSummary_->getOrCreateSetForNode(callNode, lambda);

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
      const auto targetModRefSet =
          ModRefSummary_->getOrCreateSetForNode(calleeLambda, calleeLambda);
      AddModRefSimpleConstraint(targetModRefSet, callModRef);
    }
    else if (kind == PointsToGraph::NodeKind::ImportNode)
    {
      ModRefSummary_->markSetAsCallingExternalFunction(callModRef);
    }
  }
  if (pointsToGraph.isTargetingAllExternallyAvailable(targetPtgNode))
  {
    ModRefSummary_->markSetAsCallingExternalFunction(callModRef);
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

    const RegionAwareModRefSet & fromSet = ModRefSummary_->getModRefSet(workItem);

    // Handle all simple constraints workItem -> target
    for (auto target : Context_->ModRefSetSimpleConstraints[workItem].Items())
    {
      RegionAwareModRefSet & targetSet = ModRefSummary_->getModRefSet(target);

      // Propagate flags first, to enable skipping of doubled-up memory nodes
      bool changed = targetSet.propagateFlags(fromSet);

      if (auto blocklist = Context_->ModRefSetBlocklists.find(target);
          blocklist != Context_->ModRefSetBlocklists.end())
      {
        // The target has a blocklist, avoid propagating blocked memory nodes
        for (auto [memoryNode, mayMod] : fromSet.getModRefNodes())
        {
          if (blocklist->second->Contains(memoryNode))
            continue;

          changed |= ModRefSummary_->addMemoryNodeToSet(target, memoryNode, mayMod);
        }
      }
      else
      {
        // The target does not have a blocklist, so propagate everything
        for (auto [memoryNode, mayMod] : fromSet.getModRefNodes())
        {
          changed |= ModRefSummary_->addMemoryNodeToSet(target, memoryNode, mayMod);
        }
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
    for (auto [memoryNode, _] : ModRefSummary_->getModRefSet(index).getModRefNodes())
    {
      if (blocklist->Contains(memoryNode))
        return false;
    }
  }
  return true;
}

void
RegionAwareModRefSummarizer::materializeSets()
{
  for (auto & functions : Context_->SccFunctions)
  {
    for (auto function : functions.Items())
    {
      materializeSetsInFunction(*function);
    }
  }
}

void
RegionAwareModRefSummarizer::materializeSetsInFunction(const rvsdg::LambdaNode & lambda)
{
  const auto & pointsToGraph = ModRefSummary_->GetPointsToGraph();
  // The ModRefSet representing everying that can be modified from external functions
  const auto & externModRefNodes =
      ModRefSummary_->getModRefSet(ModRefSummary_->getExternModRefSet()).getModRefNodes();

  // Only memory nodes that appear in ModRefSets without the external memory node should be kept
  util::HashSet<PointsToGraph::NodeIndex> keepMemoryNodes;

  // Among memory nodes that should be kept, the ones flagged externally available are added here.
  // When materializing sets, the flags are turned into explicit targets using this list
  std::vector<PointsToGraph::NodeIndex> materializeExternallyAvailable;
  // When a memory node we need to keep is not externally available, yet is in the
  // ModRefSet representing extern functions, the memory node is added to this list.
  // It gets materialized in all ModRefSets flagged as possibly calling external functions.
  std::vector<std::pair<PointsToGraph::NodeIndex, ModRefEffect>> materializeFromCallToExtern;

  const auto markToKeep = [&](PointsToGraph::NodeIndex memoryNode)
  {
    bool inserted = keepMemoryNodes.insert(memoryNode);
    if (!inserted)
      return;

    if (pointsToGraph.isExternallyAvailable(memoryNode))
      materializeExternallyAvailable.push_back(memoryNode);
    else if (auto it = externModRefNodes.find(memoryNode); it != externModRefNodes.end())
      materializeFromCallToExtern.push_back(*it);
  };

  markToKeep(aa::PointsToGraph::externalMemoryNode);

  // Go over all ModRefSets in the function twice
  // The first pass determines which memory nodes to keep
  const auto & allModRefSets = ModRefSummary_->getAllSetsInFunction(lambda);
  for (auto modRefSetIndex : allModRefSets)
  {
    const auto & modRefSet = ModRefSummary_->getModRefSet(modRefSetIndex);
    Context_->numModRefSetsMaterialized++;
    Context_->modRefSetSizeBeforeMaterialization += modRefSet.getModRefNodes().size();

    auto effectOnExternalNode = modRefSet.getImplicitModRefEffectForExternal(std::nullopt);

    // If this ModRefSet may both reference and modify the external memory node,
    // it will not disqualify any other memory nodes from compression
    if (effectOnExternalNode == ModRefEffect::ModRef)
      continue;

    for (auto [memoryNode, modRefEffect] : modRefSet.getModRefNodes())
    {
      JLM_ASSERT(modRefEffect != ModRefEffect::NoEffect);

      // If the set has an effect on the memory node that it does not have on the external node,
      // the memory node is disqualified from compression
      if (!isEffectSubset(modRefEffect, effectOnExternalNode))
        markToKeep(memoryNode);
    }
  }

  // Now go over all sets again, removing all memory nodes that are not on the keep list
  // and materializing memory nodes that were previously only implicit
  for (auto modRefSetIndex : allModRefSets)
  {
    auto & modRefSet = ModRefSummary_->getModRefSet(modRefSetIndex);
    modRefSet.keepSubsetOfExplicitMemoryNodes(keepMemoryNodes);
    Context_->modRefSetSizeAfterFiltering += modRefSet.getModRefNodes().size();

    // Check what effects are implicitly encoded for externally available memory nodes.
    // The flags have a minimum size, so large memory nodes may have more effects than small ones.
    // Setting an unknown size gives the largest possible effect set.
    auto effectOnAllExternalNodes = modRefSet.getImplicitModRefEffectForExternal(1);
    auto effectOnLargeExternalNodes = modRefSet.getImplicitModRefEffectForExternal(std::nullopt);
    JLM_ASSERT(isEffectSubset(effectOnAllExternalNodes, effectOnLargeExternalNodes));

    if (effectOnLargeExternalNodes == ModRefEffect::NoEffect)
    {
      // For nodes that do not have any effect on externally available memory,
      // we do not need to do any materialization.
      Context_->modRefSetSizeAfterMaterialization += modRefSet.getModRefNodes().size();
      continue;
    }

    // If small and large external memory nodes have the same effects,
    // we can materialize all external memory nodes with that effect.
    if (effectOnLargeExternalNodes == effectOnAllExternalNodes)
    {
      for (auto memoryNode : materializeExternallyAvailable)
      {
        modRefSet.addExplicitMemoryNode(memoryNode, effectOnLargeExternalNodes);
      }
    }
    else
    {
      // Materialize each memory node with the appropriate effects based on its size
      for (auto memoryNode : materializeExternallyAvailable)
      {
        const auto memoryNodeSize = pointsToGraph.tryGetNodeSize(memoryNode);
        const auto modRefEffect = modRefSet.getImplicitModRefEffectForExternal(memoryNodeSize);
        modRefSet.addExplicitMemoryNode(memoryNode, modRefEffect);
      }
    }

    // If this ModRefSet is not only accessing everything that is externally available,
    // but also possibly calling external functions, materialize from the call to external set
    if (modRefSet.mayCallExternalFunction())
    {
      Context_->numModRefSetsCallingExternalFunction++;
      for (auto [memoryNode, modRefEffect] : materializeFromCallToExtern)
      {
        modRefSet.addExplicitMemoryNode(memoryNode, modRefEffect);
      }
    }

    Context_->numModRefSetsWithEffectOnExternal++;
    Context_->modRefSetSizeAfterMaterialization += modRefSet.getModRefNodes().size();
  }
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
  std::ostringstream ss;

  ss << "ExternModRefSet: "
     << modRefSummary.getModRefSetDebugString(modRefSummary.getExternModRefSet()) << std::endl;

  auto indent = [&](size_t depth, char c = '-')
  {
    for (size_t i = 0; i < depth; i++)
      ss << c;
  };

  std::function<void(const rvsdg::Node &, size_t)> toRegionTree =
      [&](const rvsdg::Node & node, size_t depth)
  {
    // Simple nodes with no ModRefSet can be ignored
    if (dynamic_cast<const rvsdg::SimpleNode *>(&node) && !modRefSummary.hasSetForNode(node))
      return;

    indent(depth, '-');
    ss << "node " << node.DebugString() << " NodeID: " << node.GetNodeId() << ": ";
    if (modRefSummary.hasSetForNode(node))
    {
      auto modRefIndex = modRefSummary.getSetForNode(node);
      ss << modRefSummary.getModRefSetDebugString(modRefIndex) << std::endl;
    }

    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & region : structuralNode->Subregions())
      {
        indent(depth + 1, '-');
        ss << "RegionID: " << region.getRegionId() << std::endl;
        for (auto & n : region.Nodes())
          toRegionTree(n, depth + 2);
      }
    }
  };

  ss << "RootRegion:" << std::endl;
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
