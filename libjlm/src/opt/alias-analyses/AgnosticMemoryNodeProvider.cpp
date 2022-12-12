/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>

namespace jlm::aa
{

/** \brief Memory Node Provisioning of agnostic memory node provider
 *
 */
class AgnosticMemoryNodeProvisioning final : public MemoryNodeProvisioning
{
public:
  ~AgnosticMemoryNodeProvisioning() noexcept override = default;

private:
  explicit
  AgnosticMemoryNodeProvisioning(const PointsToGraph & pointsToGraph)
    : PointsToGraph_(pointsToGraph)
  {}

public:
  AgnosticMemoryNodeProvisioning(const AgnosticMemoryNodeProvisioning&) = delete;

  AgnosticMemoryNodeProvisioning(AgnosticMemoryNodeProvisioning&&) = delete;

  AgnosticMemoryNodeProvisioning &
  operator=(const AgnosticMemoryNodeProvisioning&) = delete;

  AgnosticMemoryNodeProvisioning &
  operator=(AgnosticMemoryNodeProvisioning&&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionEntryNodes(const jive::region & region) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionExitNodes(const jive::region & region) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetCallEntryNodes(const CallNode & callNode) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetCallExitNodes(const CallNode & callNode) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] HashSet<const PointsToGraph::MemoryNode*>
  GetOutputNodes(const jive::output & output) const override
  {
    JLM_ASSERT(is<PointerType>(output.type()));
    auto & registerNode = PointsToGraph_.GetRegisterNode(output);

    HashSet<const PointsToGraph::MemoryNode*> memoryNodes;
    for (auto & memoryNode : registerNode.Targets())
      memoryNodes.Insert(&memoryNode);

    return memoryNodes;
  }

  void
  SetMemoryNodes(HashSet<const PointsToGraph::MemoryNode*> memoryNodes)
  {
    MemoryNodes_ = std::move(memoryNodes);
  }

  static std::unique_ptr<AgnosticMemoryNodeProvisioning>
  Create(const PointsToGraph & pointsToGraph)
  {
    return std::unique_ptr<AgnosticMemoryNodeProvisioning>(
      new AgnosticMemoryNodeProvisioning(pointsToGraph));
  }

private:
  const PointsToGraph & PointsToGraph_;
  HashSet<const PointsToGraph::MemoryNode*> MemoryNodes_;
};

AgnosticMemoryNodeProvider::~AgnosticMemoryNodeProvider()
= default;

std::unique_ptr<MemoryNodeProvisioning>
AgnosticMemoryNodeProvider::ProvisionMemoryNodes(
  const RvsdgModule&,
  const PointsToGraph & pointsToGraph,
  StatisticsCollector& statisticsCollector)
{
  auto provisioning = AgnosticMemoryNodeProvisioning::Create(pointsToGraph);

  auto statistics = Statistics::Create(statisticsCollector, pointsToGraph);

  statistics->StartCollecting();
  HashSet<const PointsToGraph::MemoryNode*> memoryNodes;
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
  provisioning->SetMemoryNodes(std::move(memoryNodes));
  statistics->StopCollecting();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  return provisioning;
}

std::unique_ptr<MemoryNodeProvisioning>
AgnosticMemoryNodeProvider::Create(
  const RvsdgModule & rvsdgModule,
  const PointsToGraph & pointsToGraph,
  StatisticsCollector & statisticsCollector)
{
  AgnosticMemoryNodeProvider provider;
  return provider.ProvisionMemoryNodes(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<MemoryNodeProvisioning>
AgnosticMemoryNodeProvider::Create(
  const RvsdgModule & rvsdgModule,
  const PointsToGraph & pointsToGraph)
{
  StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

}