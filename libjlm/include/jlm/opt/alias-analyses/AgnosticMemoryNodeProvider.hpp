/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_AGNOSTICMEMORYNODEPROVIDER_HPP
#define JLM_OPT_ALIAS_ANALYSES_AGNOSTICMEMORYNODEPROVIDER_HPP

#include <jlm/opt/alias-analyses/MemoryNodeProvider.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::aa
{

class AgnosticMemoryNodeProvisioning;

/** \brief Agnostic memory node provider
 *
 * The key idea of the agnostic memory node provider is that \b all memory states are routed through \b all
 * structural nodes irregardless of whether these states are required by any simple nodes within the structural nodes.
 * This strategy ensures that the state of a memory location is always present for encoding while avoiding the
 * complexity of an additional analysis for determining the required routing path of the states. The drawback is that
 * a lot of states are routed through structural nodes where they are not needed, potentially leading to a significant
 * runtime of the encoder for bigger RVSDGs.
 *
 * @see MemoryNodeProvider
 * @see MemoryStateEncoder
 */
class AgnosticMemoryNodeProvider final : public MemoryNodeProvider
{
public:
  class Statistics;

  ~AgnosticMemoryNodeProvider() override;

  AgnosticMemoryNodeProvider() = default;

  AgnosticMemoryNodeProvider(const AgnosticMemoryNodeProvider&) = delete;

  AgnosticMemoryNodeProvider(AgnosticMemoryNodeProvider&&) = delete;

  AgnosticMemoryNodeProvider &
  operator=(const AgnosticMemoryNodeProvider&) = delete;

  AgnosticMemoryNodeProvider &
  operator=(AgnosticMemoryNodeProvider&&) = delete;

  void
  ProvisionMemoryNodes(
    const RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    StatisticsCollector & statisticsCollector) override;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override;

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionEntryNodes(const jive::region & region) const override;

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionExitNodes(const jive::region & region) const override;

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetCallEntryNodes(const CallNode & callNode) const override;

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetCallExitNodes(const CallNode & callNode) const override;

  [[nodiscard]] HashSet<const PointsToGraph::MemoryNode*>
  GetOutputNodes(const jive::output & output) const override;

  /**
   * Creates a AgnosticMemoryNodeProvider and calls the ProvisionMemoryNodes() method.
   *
   * @param rvsdgModule The RVSDG module on which the provision should be performed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return A new instance of AgnosticMemoryNodeProvider.
   */
  static std::unique_ptr<AgnosticMemoryNodeProvider>
  Create(
    const RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    StatisticsCollector & statisticsCollector);

  /**
   * Creates a AgnosticMemoryNodeProvider and calls the ProvisionMemoryNodes() method.
   *
   * @param rvsdgModule The RVSDG module on which the provision should be performed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   *
   * @return A new instance of AgnosticMemoryNodeProvider.
   */
  static std::unique_ptr<AgnosticMemoryNodeProvider>
  Create(
    const RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph);

private:
  std::unique_ptr<AgnosticMemoryNodeProvisioning> Provisioning_;
};

/** \brief Agnostic memory node provider statistics
 *
 * The statistics collected when running the agnostic memory node provider.
 *
 * @See AgnosticMemoryNodeProvider
 */
class AgnosticMemoryNodeProvider::Statistics final : public jlm::Statistics
{
public:
  Statistics(
    const StatisticsCollector & statisticsCollector,
    const PointsToGraph & pointsToGraph)
    : jlm::Statistics(Statistics::Id::MemoryNodeProvisioning)
    , NumPointsToGraphMemoryNodes_(0)
    , StatisticsCollector_(statisticsCollector)
  {
    if (!StatisticsCollector_.IsDemanded(*this))
      return;

    NumPointsToGraphMemoryNodes_ = pointsToGraph.NumMemoryNodes();
  }

  [[nodiscard]] size_t
  NumPointsToGraphMemoryNodes() const noexcept
  {
    return NumPointsToGraphMemoryNodes_;
  }

  [[nodiscard]] size_t
  GetTime() const noexcept
  {
    return Timer_.ns();
  }

  void
  StartCollecting() noexcept
  {
    if (!StatisticsCollector_.IsDemanded(*this))
      return;

    Timer_.start();
  }

  void
  StopCollecting() noexcept
  {
    if (!StatisticsCollector_.IsDemanded(*this))
      return;

    Timer_.stop();
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return strfmt(
      "AgnosticMemoryNodeProvision ",
      "#PointsToGraphMemoryNodes:", NumPointsToGraphMemoryNodes_, " ",
      "Time[ns]:", Timer_.ns()
      );
  }

  static std::unique_ptr<Statistics>
  Create(
    const StatisticsCollector & statisticsCollector,
    const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Statistics>(statisticsCollector, pointsToGraph);
  }

private:
  jlm::timer Timer_;
  size_t NumPointsToGraphMemoryNodes_;
  const StatisticsCollector & StatisticsCollector_;
};

}

#endif //JLM_OPT_ALIAS_ANALYSES_AGNOSTICMEMORYNODEPROVIDER_HPP
