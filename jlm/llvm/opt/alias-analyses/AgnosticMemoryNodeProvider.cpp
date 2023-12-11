/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>

namespace jlm::llvm::aa
{

/** \brief Memory Node Provisioning of agnostic memory node provider
 *
 */
class AgnosticMemoryNodeProvisioning final : public MemoryNodeProvisioning
{
public:
  ~AgnosticMemoryNodeProvisioning() noexcept override = default;

private:
  AgnosticMemoryNodeProvisioning(
      const PointsToGraph & pointsToGraph,
      util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes)
      : PointsToGraph_(pointsToGraph),
        MemoryNodes_(std::move(memoryNodes))
  {}

public:
  AgnosticMemoryNodeProvisioning(const AgnosticMemoryNodeProvisioning &) = delete;

  AgnosticMemoryNodeProvisioning(AgnosticMemoryNodeProvisioning &&) = delete;

  AgnosticMemoryNodeProvisioning &
  operator=(const AgnosticMemoryNodeProvisioning &) = delete;

  AgnosticMemoryNodeProvisioning &
  operator=(AgnosticMemoryNodeProvisioning &&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionEntryNodes(const rvsdg::region & region) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionExitNodes(const rvsdg::region & region) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetCallEntryNodes(const CallNode & callNode) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetCallExitNodes(const CallNode & callNode) const override
  {
    return MemoryNodes_;
  }

  [[nodiscard]] util::HashSet<const PointsToGraph::MemoryNode *>
  GetOutputNodes(const rvsdg::output & output) const override
  {
    JLM_ASSERT(is<PointerType>(output.type()));

    util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes;
    if (GetOutputNodesFromRegisterNode(output, memoryNodes))
      return memoryNodes;

    if (GetOutputNodesFromRegisterSetNode(output, memoryNodes))
      return memoryNodes;

    throw util::error("Cannot find register in points-to graph.");
  }

  static std::unique_ptr<AgnosticMemoryNodeProvisioning>
  Create(
      const PointsToGraph & pointsToGraph,
      util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes)
  {
    return std::unique_ptr<AgnosticMemoryNodeProvisioning>(
        new AgnosticMemoryNodeProvisioning(pointsToGraph, std::move(memoryNodes)));
  }

private:
  [[nodiscard]] bool
  GetOutputNodesFromRegisterNode(
      const rvsdg::output & output,
      util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes) const
  {
    const PointsToGraph::RegisterNode * registerNode;
    try
    {
      registerNode = &PointsToGraph_.GetRegisterNode(output);
    }
    catch (...)
    {
      return false;
    }

    for (auto & memoryNode : registerNode->Targets())
      memoryNodes.Insert(&memoryNode);

    return true;
  }

  [[nodiscard]] bool
  GetOutputNodesFromRegisterSetNode(
      const rvsdg::output & output,
      util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes) const
  {
    const PointsToGraph::RegisterSetNode * registerSetNode;
    try
    {
      registerSetNode = &PointsToGraph_.GetRegisterSetNode(output);
    }
    catch (...)
    {
      return false;
    }

    for (auto & memoryNode : registerSetNode->Targets())
      memoryNodes.Insert(&memoryNode);

    return true;
  }

  const PointsToGraph & PointsToGraph_;
  util::HashSet<const PointsToGraph::MemoryNode *> MemoryNodes_;
};

AgnosticMemoryNodeProvider::~AgnosticMemoryNodeProvider() = default;

std::unique_ptr<MemoryNodeProvisioning>
AgnosticMemoryNodeProvider::ProvisionMemoryNodes(
    const RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics =
      Statistics::Create(rvsdgModule.SourceFileName(), statisticsCollector, pointsToGraph);
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

  auto provisioning = AgnosticMemoryNodeProvisioning::Create(pointsToGraph, std::move(memoryNodes));

  statistics->StopCollecting();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  return provisioning;
}

std::unique_ptr<MemoryNodeProvisioning>
AgnosticMemoryNodeProvider::Create(
    const RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  AgnosticMemoryNodeProvider provider;
  return provider.ProvisionMemoryNodes(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<MemoryNodeProvisioning>
AgnosticMemoryNodeProvider::Create(
    const RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph)
{
  util::StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

}
