/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_AGNOSTICMEMORYNODEPROVIDER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_AGNOSTICMEMORYNODEPROVIDER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm::aa
{

/** \brief Agnostic memory node provider
 *
 * The key idea of the agnostic memory node provider is that \b all memory states are routed through
 * \b all structural nodes irregardless of whether these states are required by any simple nodes
 * within the structural nodes. This strategy ensures that the state of a memory location is always
 * present for encoding while avoiding the complexity of an additional analysis for determining the
 * required routing path of the states. The drawback is that a lot of states are routed through
 * structural nodes where they are not needed, potentially leading to a significant runtime of the
 * encoder for bigger RVSDGs.
 *
 * @see MemoryNodeProvider
 * @see MemoryStateEncoder
 */
class AgnosticMemoryNodeProvider final : public ModRefSummarizer
{
public:
  class Statistics;

  ~AgnosticMemoryNodeProvider() override;

  AgnosticMemoryNodeProvider() = default;

  AgnosticMemoryNodeProvider(const AgnosticMemoryNodeProvider &) = delete;

  AgnosticMemoryNodeProvider(AgnosticMemoryNodeProvider &&) = delete;

  AgnosticMemoryNodeProvider &
  operator=(const AgnosticMemoryNodeProvider &) = delete;

  AgnosticMemoryNodeProvider &
  operator=(AgnosticMemoryNodeProvider &&) = delete;

  std::unique_ptr<ModRefSummary>
  SummarizeModRefs(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector) override;

  /**
   * Creates a AgnosticMemoryNodeProvider and calls the ProvisionMemoryNodes() method.
   *
   * @param rvsdgModule The RVSDG module on which the provision should be performed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return A new instance of MemoryNodeProvisioning.
   */
  static std::unique_ptr<ModRefSummary>
  Create(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector);

  /**
   * Creates a AgnosticMemoryNodeProvider and calls the ProvisionMemoryNodes() method.
   *
   * @param rvsdgModule The RVSDG module on which the provision should be performed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   *
   * @return A new instance of MemoryNodeProvisioning.
   */
  static std::unique_ptr<ModRefSummary>
  Create(const rvsdg::RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph);
};

/** \brief Agnostic memory node provider statistics
 *
 * The statistics collected when running the agnostic memory node provider.
 *
 * @see AgnosticMemoryNodeProvider
 */
class AgnosticMemoryNodeProvider::Statistics final : public util::Statistics
{
public:
  Statistics(
      const util::filepath & sourceFile,
      const util::StatisticsCollector & statisticsCollector,
      const PointsToGraph & pointsToGraph)
      : util::Statistics(Statistics::Id::AgnosticMemoryNodeProvisioning, sourceFile),
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
      const util::filepath & sourceFile,
      const util::StatisticsCollector & statisticsCollector,
      const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Statistics>(sourceFile, statisticsCollector, pointsToGraph);
  }

private:
  const util::StatisticsCollector & StatisticsCollector_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_AGNOSTICMEMORYNODEPROVIDER_HPP
