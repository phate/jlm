/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMEMORYNODEPROVIDER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMEMORYNODEPROVIDER_HPP

#include <jlm/llvm/opt/alias-analyses/MemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm::aa
{

class RegionAwareMemoryNodeProvisioning;

/** \brief Region-aware memory node provider
 *
 * The key idea of the region-aware memory node provider is to only provide memory locations for a
 * structural node that are actually utilized within its regions. This ensures that no superfluous
 * states will be routed through structural nodes and renders them independent if they do not
 * reference the same memory location. The region-aware analysis proceeds as follows:
 *
 * 1. Annotation: Each region is annotated with its utilized memory locations, the contained
 * structural nodes, the function calls it contains, and the RVSDG nodes that reference unknown
 * memory locations.
 *
 * 2. Propagation: The memory locations and RVSDG nodes that reference unknown memory locations are
 * propagated through the graph such that a region always has the same memory locations and RVSDG
 * nodes annotated as its contained structural nodes and function calls.
 *
 * 3. Resolution of unknown memory locations: The unknown memory location references are resolved by
 * annotating the regions of the corresponding RVSDG nodes with all the memory locations that are
 * referenced before and after this respective RVSDG node.
 *
 * 4. Propagation: The memory locations are propagated through the graph again. After this phase, a
 * fix-point is reached and all regions are annotated with the required memory locations.
 *
 * @see MemoryNodeProvider
 * @see MemoryStateEncoder
 */
class RegionAwareMemoryNodeProvider final : public MemoryNodeProvider
{
public:
  class Statistics;

  ~RegionAwareMemoryNodeProvider() noexcept override;

  RegionAwareMemoryNodeProvider();

  RegionAwareMemoryNodeProvider(const RegionAwareMemoryNodeProvider &) = delete;

  RegionAwareMemoryNodeProvider(RegionAwareMemoryNodeProvider &&) = delete;

  RegionAwareMemoryNodeProvider &
  operator=(const RegionAwareMemoryNodeProvider &) = delete;

  RegionAwareMemoryNodeProvider &
  operator=(RegionAwareMemoryNodeProvider &&) = delete;

  std::unique_ptr<MemoryNodeProvisioning>
  ProvisionMemoryNodes(
      const RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      jlm::util::StatisticsCollector & statisticsCollector) override;

  /**
   * Creates a RegionAwareMemoryNodeProvider and calls the ProvisionMemoryNodes() method.
   *
   * @param rvsdgModule The RVSDG module on which the provision should be performed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return A new instance of MemoryNodeProvisioning.
   */
  static std::unique_ptr<MemoryNodeProvisioning>
  Create(
      const RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      jlm::util::StatisticsCollector & statisticsCollector);

  /**
   * Creates a RegionAwareMemoryNodeProvider and calls the ProvisionMemoryNodes() method.
   *
   * @param rvsdgModule The RVSDG module on which the provision should be performed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   *
   * @return A new instance of MemoryNodeProvisioning.
   */
  static std::unique_ptr<MemoryNodeProvisioning>
  Create(const RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph);

private:
  /**
   * Annotates a region with the memory locations utilized by the contained simple RVSDG nodes,
   * e.g., load, store, etc. nodes, the contained function calls, and the simple RVSDG nodes that
   * reference unknown memory locations.
   *
   * The annotation phase starts at the RVSDG root region and simply iterates through all the nodes
   * within a region and performs the appropriate action for a node. It recursively traverses the
   * subregions of structural nodes until all nodes within all regions of the graph have been
   * visited.
   *
   * @param region The to be annotated region.
   */
  void
  AnnotateRegion(jlm::rvsdg::region & region);

  void
  AnnotateSimpleNode(const jlm::rvsdg::simple_node & provider);

  void
  AnnotateStructuralNode(const jlm::rvsdg::structural_node & structuralNode);

  void
  AnnotateLoad(const LoadNode & loadNode);

  void
  AnnotateStore(const StoreNode & storeNode);

  void
  AnnotateAlloca(const jlm::rvsdg::simple_node & allocaNode);

  void
  AnnotateMalloc(const jlm::rvsdg::simple_node & mallocNode);

  void
  AnnotateFree(const jlm::rvsdg::simple_node & freeNode);

  void
  AnnotateCall(const CallNode & callNode);

  void
  AnnotateMemcpy(const jlm::rvsdg::simple_node & memcpyNode);

  /**
   *  Propagates the utilized memory locations and simple RVSDG nodes that reference unknown memory
   * locations through the graph such that a region always contains the memory locations and simple
   * RVSDG nodes of its contained structural nodes and function calls.
   *
   *  The propagation phase traverses the call graph in the RVSDG root region top-down, i.e. a
   * lambda node is always visited before its function calls. For each lambda node, it propagates
   * the memory locations and simple RVSDG nodes from the innermost subregions outward to the lambda
   * region. Moreover, it retrieves the locations and nodes for direct non-recursive calls from the
   * respective lambda nodes, which have already been visited. This ensures that all required memory
   * locations that are referenced within the lambda and all simple RVSDG nodes are annotated on the
   * lambda region. For phi nodes, it propagates the entities similarly as for non-recursive lambda
   *  nodes, but avoids to follow direct recursive calls to other lambda nodes within the phi
   * region. After each lambda node of a phi node is handled, the union of the memory locations as
   * well as the union of the simple RVSDG nodes of all lambdas in the phi node are computed, and
   * associated with each lambda in the phi node.
   *
   * @param rvsdgModule The RVSDG module on which the propagation is performed.
   *
   * @see ExtractLambdaNodes()
   */
  void
  Propagate(const RvsdgModule & rvsdgModule);

  void
  PropagateRegion(const jlm::rvsdg::region & region);

  void
  PropagatePhi(const phi::node & phiNode);

  /**
   * Resolves all references to unknown memory locations.
   *
   * After the propagation phase, the tail lambda regions contain all memory locations and simple
   * RVSDG nodes that reference unknown memory locations from any of the dependent lambdas. This
   * phase simply iterates through these simple nodes and adds the memory locations of the tail
   * lambda region to the region of the simple RVSDG node. The memory locations of the tail lambda
   * region are all memory locations that are referenced before and after a simple RVSDG node. By
   * adding these memory locations to the respective region of the simple RVSDG nodes, we ensure
   * that all memory locations from before and after these nodes are available for encoding.
   *
   * @param rvsdgModule The RVSDG module for which to resolve the unknown memory location
   * references.
   *
   * @see ExtractRvsdgTailNodes()
   */
  void
  ResolveUnknownMemoryNodeReferences(const RvsdgModule & rvsdgModule);

  std::unique_ptr<RegionAwareMemoryNodeProvisioning> Provisioning_;
};

/** \brief Region-aware memory node provider statistics
 *
 * The statistics collected when running the region-aware memory node provider.
 *
 * @see RegionAwareMemoryNodeProvider
 */
class RegionAwareMemoryNodeProvider::Statistics final : public util::Statistics
{
  const char * NumRvsdgRegionsLabel_ = "#RvsdgRegions";

  const char * AnnotationTimerLabel_ = "AnnotationTime";
  const char * PropagationPass1TimerLabel_ = "PropagationPass1Time";
  const char * PropagationPass2TimerLabel_ = "PropagationPass2Time";
  const char * ResolveUnknownMemoryReferenceTimerLabel_ = "ResolveUnknownMemoryReferenceTime";

public:
  ~Statistics() override = default;

  explicit Statistics(
      const util::StatisticsCollector & statisticsCollector,
      const RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph)
      : util::Statistics(
          Statistics::Id::RegionAwareMemoryNodeProvisioning,
          rvsdgModule.SourceFileName()),
        StatisticsCollector_(statisticsCollector)
  {
    if (!IsDemanded())
      return;

    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(rvsdgModule.Rvsdg().root()));
    AddMeasurement(NumRvsdgRegionsLabel_, rvsdg::region::NumRegions(*rvsdgModule.Rvsdg().root()));
    AddMeasurement(Label::NumPointsToGraphMemoryNodes, pointsToGraph.NumMemoryNodes());
  }

  [[nodiscard]] size_t
  NumRvsdgNodes() const noexcept
  {
    return GetMeasurementValue<uint64_t>(Label::NumRvsdgNodes);
  }

  [[nodiscard]] size_t
  NumRvsdgRegions() const noexcept
  {
    return GetMeasurementValue<uint64_t>(NumRvsdgRegionsLabel_);
  }

  [[nodiscard]] size_t
  NumPointsToGraphMemoryNodes() const noexcept
  {
    return GetMeasurementValue<uint64_t>(Label::NumPointsToGraphMemoryNodes);
  }

  [[nodiscard]] size_t
  GetAnnotationStatisticsTime() const noexcept
  {
    return GetTimer(AnnotationTimerLabel_).ns();
  }

  [[nodiscard]] size_t
  GetPropagationPass1Time() const noexcept
  {
    return GetTimer(PropagationPass1TimerLabel_).ns();
  }

  [[nodiscard]] size_t
  GetPropagationPass2Time() const noexcept
  {
    return GetTimer(PropagationPass2TimerLabel_).ns();
  }

  [[nodiscard]] size_t
  GetResolveUnknownMemoryNodeReferencesTime() const noexcept
  {
    return GetTimer(ResolveUnknownMemoryReferenceTimerLabel_).ns();
  }

  void
  StartAnnotationStatistics() noexcept
  {
    if (!IsDemanded())
      return;

    AddTimer(AnnotationTimerLabel_).start();
  }

  void
  StopAnnotationStatistics() noexcept
  {
    if (!IsDemanded())
      return;

    GetTimer(AnnotationTimerLabel_).stop();
  }

  void
  StartPropagationPass1Statistics() noexcept
  {
    if (!IsDemanded())
      return;

    AddTimer(PropagationPass1TimerLabel_).start();
  }

  void
  StopPropagationPass1Statistics() noexcept
  {
    if (!IsDemanded())
      return;

    GetTimer(PropagationPass1TimerLabel_).stop();
  }

  void
  StartResolveUnknownMemoryNodeReferencesStatistics() noexcept
  {
    if (!IsDemanded())
      return;

    AddTimer(ResolveUnknownMemoryReferenceTimerLabel_).start();
  }

  void
  StopResolveUnknownMemoryNodeReferencesStatistics() noexcept
  {
    if (!IsDemanded())
      return;

    GetTimer(ResolveUnknownMemoryReferenceTimerLabel_).stop();
  }

  void
  StartPropagationPass2Statistics() noexcept
  {
    if (!IsDemanded())
      return;

    AddTimer(PropagationPass2TimerLabel_).start();
  }

  void
  StopPropagationPass2Statistics() noexcept
  {
    if (!IsDemanded())
      return;

    GetTimer(PropagationPass2TimerLabel_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(
      const util::StatisticsCollector & statisticsCollector,
      const RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Statistics>(statisticsCollector, rvsdgModule, pointsToGraph);
  }

private:
  /**
   * Checks if the pass statistics are demanded.
   *
   * @return True if the pass statistic is demanded, otherwise false.
   */
  [[nodiscard]] bool
  IsDemanded() const noexcept
  {
    return StatisticsCollector_.GetSettings().IsDemanded(GetId());
  }

  const util::StatisticsCollector & StatisticsCollector_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMEMORYNODEPROVIDER_HPP
