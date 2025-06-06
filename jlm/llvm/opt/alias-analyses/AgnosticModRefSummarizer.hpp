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

  AgnosticModRefSummarizer() = default;

  AgnosticModRefSummarizer(const AgnosticModRefSummarizer &) = delete;

  AgnosticModRefSummarizer(AgnosticModRefSummarizer &&) = delete;

  AgnosticModRefSummarizer &
  operator=(const AgnosticModRefSummarizer &) = delete;

  AgnosticModRefSummarizer &
  operator=(AgnosticModRefSummarizer &&) = delete;

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
