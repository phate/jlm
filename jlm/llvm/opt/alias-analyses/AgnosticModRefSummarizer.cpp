/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>

namespace jlm::llvm::aa
{

/** \brief Mod/Ref summary of agnostic mod/ref summarizer
 *
 */
class AgnosticModRefSummary final : public ModRefSummary
{
public:
  ~AgnosticModRefSummary() noexcept override = default;

private:
  AgnosticModRefSummary(
      const PointsToGraph & pointsToGraph,
      util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes)
      : PointsToGraph_(pointsToGraph),
        MemoryNodes_(std::move(memoryNodes))
  {}

public:
  AgnosticModRefSummary(const AgnosticModRefSummary &) = delete;

  AgnosticModRefSummary(AgnosticModRefSummary &&) = delete;

  AgnosticModRefSummary &
  operator=(const AgnosticModRefSummary &) = delete;

  AgnosticModRefSummary &
  operator=(AgnosticModRefSummary &&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetSimpleNodeModRef([[maybe_unused]] const rvsdg::SimpleNode & node) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetGammaEntryModRef([[maybe_unused]] const rvsdg::GammaNode & gamma) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetGammaExitModRef([[maybe_unused]] const rvsdg::GammaNode & gamma) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetThetaModRef([[maybe_unused]] const rvsdg::ThetaNode & theta) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaEntryModRef([[maybe_unused]]const rvsdg::LambdaNode & lambda) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaExitModRef([[maybe_unused]] const rvsdg::LambdaNode & lambda) const override
  {
    return MemoryNodes_;
  }

  static std::unique_ptr<AgnosticModRefSummary>
  Create(
      const PointsToGraph & pointsToGraph,
      util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes)
  {
    return std::unique_ptr<AgnosticModRefSummary>(
        new AgnosticModRefSummary(pointsToGraph, std::move(memoryNodes)));
  }

private:
  const PointsToGraph & PointsToGraph_;
  util::HashSet<const PointsToGraph::MemoryNode *> MemoryNodes_;
};

AgnosticModRefSummarizer::~AgnosticModRefSummarizer() = default;

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::SummarizeModRefs(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics =
      Statistics::Create(rvsdgModule.SourceFilePath().value(), statisticsCollector, pointsToGraph);
  statistics->StartCollecting();

  util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes;
  for (auto & allocaNode : pointsToGraph.AllocaNodes())
    memoryNodes.Insert(&allocaNode);

  for (auto & deltaNode : pointsToGraph.DeltaNodes())
    memoryNodes.Insert(&deltaNode);

  for (auto & lambdaNode : pointsToGraph.LambdaNodes())
    memoryNodes.Insert(&lambdaNode);

  for (auto & mallocNode : pointsToGraph.MallocNodes())
    memoryNodes.Insert(&mallocNode);

  for (auto & importNode : pointsToGraph.ImportNodes())
    memoryNodes.Insert(&importNode);

  memoryNodes.Insert(&pointsToGraph.GetExternalMemoryNode());

  auto modRefSummary = AgnosticModRefSummary::Create(pointsToGraph, std::move(memoryNodes));

  statistics->StopCollecting();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  return modRefSummary;
}

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  AgnosticModRefSummarizer summarizer;
  return summarizer.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph)
{
  util::StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

}
