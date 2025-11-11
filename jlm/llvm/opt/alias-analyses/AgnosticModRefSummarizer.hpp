/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_AGNOSTICMODREFSUMMARIZER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_AGNOSTICMODREFSUMMARIZER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm::aa
{

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
  [[nodiscard]] static util::HashSet<const PointsToGraph::MemoryNode *>
  GetAllMemoryNodes(const PointsToGraph & pointsToGraph);

  /**
   * Helper for adding all memory nodes the given \p output may target to a Mod/Ref set
   * @param output the pointer typed output
   * @param modRefSet the set of memory nodes that should be expanded with \p output's targets
   */
  void
  AddPointerTargetsToModRefSet(
      const rvsdg::Output & output,
      util::HashSet<const PointsToGraph::MemoryNode *> & modRefSet) const;

  /**
   * Recursively traverses the given \p region, creating Mod/Ref sets for simple nodes.
   * @param region the region to traverse
   */
  void
  AnnotateRegion(const rvsdg::Region & region);

  /**
   * Creates a Mod/Ref set for the given simple node if it belongs in the Mod/Ref set map.
   * Only nodes that affect memory are given Mod/Ref sets.
   * \ref CallOperations are not included, as the agnostic summary assumes calls touch everything.
   * @param node the simple node
   */
  void
  AnnotateSimpleNode(const rvsdg::SimpleNode & node);

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

    AddMeasurement(Label::NumPointsToGraphMemoryNodes, pointsToGraph.NumMemoryNodes());
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
