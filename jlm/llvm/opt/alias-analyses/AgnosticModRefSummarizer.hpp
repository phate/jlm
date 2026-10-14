/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_AGNOSTICMODREFSUMMARIZER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_AGNOSTICMODREFSUMMARIZER_HPP

#include "jlm/rvsdg/lambda.hpp"
#include "jlm/rvsdg/region.hpp"
#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>
#include <optional>

namespace jlm::llvm::aa
{

class AgnosticModRefSet;
class AgnosticModRefSummary;

/** \brief Agnostic mod/ref summarizer
 *
 * The key idea of the agnostic mod/ref summarizer is that \b all memory states are routed through
 * \b all structural nodes regardless of whether these states are required by any simple nodes
 * within the structural nodes. This strategy ensures that the state of a memory location is always
 * present for encoding while avoiding the complexity of an additional analysis for determining the
 * required routing path of the states. The drawback is that a lot of states are routed through
 * structural nodes where they are not needed, potentially leading to a significant runtime of the
 * encoder for bigger RVSDGs.
 *
 * @see ModRefSummarizer
 * @see MemoryStateEncoder
 */
class AgnosticModRefSummarizer final : public ModRefSummarizer
{
public:
  class Statistics;

  ~AgnosticModRefSummarizer() override;

  AgnosticModRefSummarizer();

  AgnosticModRefSummarizer(const AgnosticModRefSummarizer &) = delete;

  AgnosticModRefSummarizer &
  operator=(const AgnosticModRefSummarizer &) = delete;

  std::unique_ptr<ModRefSummary>
  SummarizeModRefs(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector) override;

  /**
   * Creates a AgnosticModRefSummarizer and calls the SummarizeModeRefs() method.
   *
   * @param rvsdgModule The RVSDG module for which a \ref ModRefSummary should be computed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return A new instance of ModRefSummary.
   */
  static std::unique_ptr<ModRefSummary>
  Create(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector);

  /**
   * Creates a AgnosticModRefSummarizer and calls the SummarizeModRefs() method.
   *
   * @param rvsdgModule The RVSDG module for which the \ref ModRefSummary should be computed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   *
   * @return A new instance of ModRefSummary.
   */
  static std::unique_ptr<ModRefSummary>
  Create(const rvsdg::RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph);

private:
  /**
   * Creates a set containing all memory nodes in the given \p pointsToGraph
   */
  [[nodiscard]] static AgnosticModRefSet
  GetAllMemoryNodes(const PointsToGraph & pointsToGraph);

  /**
   * Helper for adding all memory nodes the given \p output may target to a \ref ModRefSet
   * @param modRefSet the set of memory nodes that should be expanded with \p output's targets
   * @param output the pointer typed output
   * @param minSize an optional minimum size of memory locations added to the modRefSet.
   * @param modRefEffect the effect that is possibly performed on the targeted memory nodes
   */
  void
  addPointerTargetsToModRefSet(
      AgnosticModRefSet & modRefSet,
      const rvsdg::Output & output,
      std::optional<size_t> minSize,
      ModRefEffect modRefEffect) const;

  /**
   * Recursively traverses the given \p region, creating Mod/Ref sets for all functions.
   * @param region the region to traverse
   */
  void
  annotateInterproceduralRegion(const rvsdg::Region & region);

  /**
   * Creates ModRefSets for the given function and the nodes it contains.
   * @param lambda the function's lambda node
   */
  void
  annotateFunction(const rvsdg::LambdaNode & lambda);

  /**
   * Recursively traverses the given \p region, creating Mod/Ref sets for nodes.
   * @param region the region to traverse
   * @param lambda the function containing the region
   * @param functionModRefSetIndex the ModRefSet representing all memory effects in the function,
   * use for the lambda itself, all structural nodes and all calls in the function
   * @param modRefSetsInFunction a vector for tracking all ModRefSets created in the function.
   */
  void
  annotateIntraproceduralRegion(
      const rvsdg::Region & region,
      size_t functionModRefSetIndex,
      std::vector<size_t> & modRefSetsInFunction);

  /**
   * Creates a Mod/Ref set for the given simple node if it belongs in the Mod/Ref set map.
   * Only nodes that affect memory are given Mod/Ref sets.
   * \ref llvm::CallOperation nodes are not included, and are instead given the function ModRefSet.
   * @param node the simple node
   * @param functionModRefSetIndex the index of the ModRefSet representing the function
   * @param modRefSetsInFunction a list for tracking all created ModRefSets in the function
   */
  void
  annotateSimpleNode(
      const rvsdg::SimpleNode & node,
      size_t functionModRefSetIndex,
      std::vector<size_t> & modRefSetsInFunction);

  // The ModRefSummary being created by this class
  std::unique_ptr<AgnosticModRefSummary> ModRefSummary_;
};

/** \brief Agnostic mod/ref summarizer statistics
 *
 * The statistics collected when running the agnostic mod/ref summarizer.
 *
 * @see AgnosticModRefSummarizer
 */
class AgnosticModRefSummarizer::Statistics final : public util::Statistics
{
  static constexpr auto memoryStateDistributionLabel_ = "MemoryStateDistribution";

public:
  Statistics(
      const util::FilePath & sourceFile,
      const util::StatisticsCollector & statisticsCollector,
      const PointsToGraph & pointsToGraph)
      : util::Statistics(Id::AgnosticModRefSummarizer, sourceFile),
        StatisticsCollector_(statisticsCollector)
  {
    if (!StatisticsCollector_.IsDemanded(*this))
      return;

    AddMeasurement(Label::NumPointsToGraphMemoryNodes, pointsToGraph.numMemoryNodes());
  }

  [[nodiscard]] size_t
  NumPointsToGraphMemoryNodes() const noexcept
  {
    return GetMeasurementValue<uint64_t>(Label::NumPointsToGraphMemoryNodes);
  }

  [[nodiscard]] size_t
  GetTime() const noexcept
  {
    return GetTimer(Label::Timer).ns();
  }

  void
  StartCollecting() noexcept
  {
    if (!StatisticsCollector_.IsDemanded(*this))
      return;

    AddTimer(Label::Timer).start();
  }

  void
  StopCollecting() noexcept
  {
    if (!StatisticsCollector_.IsDemanded(*this))
      return;

    GetTimer(Label::Timer).stop();
  }

  void
  addMemoryStateDistribution(const std::vector<MemoryStateSummary> & memoryStateDistribution)
  {
    AddMeasurement(memoryStateDistributionLabel_, toString(memoryStateDistribution));
  }

  static std::unique_ptr<Statistics>
  Create(
      const util::FilePath & sourceFile,
      const util::StatisticsCollector & statisticsCollector,
      const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Statistics>(sourceFile, statisticsCollector, pointsToGraph);
  }

private:
  const util::StatisticsCollector & StatisticsCollector_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_AGNOSTICMODREFSUMMARIZER_HPP
