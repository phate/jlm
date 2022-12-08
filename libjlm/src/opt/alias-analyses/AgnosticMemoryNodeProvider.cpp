/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::aa
{

AgnosticMemoryNodeProvider::AgnosticMemoryNodeProvider(const PointsToGraph & pointsToGraph)
  : PointsToGraph_(pointsToGraph)
{}

void
AgnosticMemoryNodeProvider::ProvisionMemoryNodes(
  const RvsdgModule&,
  StatisticsCollector&)
{
  for (auto & allocaNode : PointsToGraph_.AllocaNodes())
    MemoryNodes_.Insert(&allocaNode);

  for (auto & deltaNode : PointsToGraph_.DeltaNodes())
    MemoryNodes_.Insert(&deltaNode);

  for (auto & lambdaNode : PointsToGraph_.LambdaNodes())
    MemoryNodes_.Insert(&lambdaNode);

  for (auto & mallocNode : PointsToGraph_.MallocNodes())
    MemoryNodes_.Insert(&mallocNode);

  for (auto & importNode : PointsToGraph_.ImportNodes())
    MemoryNodes_.Insert(&importNode);

  MemoryNodes_.Insert(&PointsToGraph_.GetExternalMemoryNode());
}

std::unique_ptr<AgnosticMemoryNodeProvider>
AgnosticMemoryNodeProvider::Create(
  const RvsdgModule & rvsdgModule,
  const PointsToGraph & pointsToGraph,
  StatisticsCollector & statisticsCollector)
{
  std::unique_ptr<AgnosticMemoryNodeProvider> provider(new AgnosticMemoryNodeProvider(pointsToGraph));
  provider->ProvisionMemoryNodes(rvsdgModule, statisticsCollector);

  return provider;
}

std::unique_ptr<AgnosticMemoryNodeProvider>
AgnosticMemoryNodeProvider::Create(
  const RvsdgModule & rvsdgModule,
  const PointsToGraph & pointsToGraph)
{
  StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

const PointsToGraph &
AgnosticMemoryNodeProvider::GetPointsToGraph() const noexcept
{
  return PointsToGraph_;
}

const HashSet<const PointsToGraph::MemoryNode*> &
AgnosticMemoryNodeProvider::GetRegionEntryNodes(const jive::region&) const
{
  return MemoryNodes_;
}

const HashSet<const PointsToGraph::MemoryNode*> &
AgnosticMemoryNodeProvider::GetRegionExitNodes(const jive::region&) const
{
  return MemoryNodes_;
}

const HashSet<const PointsToGraph::MemoryNode*> &
AgnosticMemoryNodeProvider::GetCallEntryNodes(const CallNode&) const
{
  return MemoryNodes_;
}

const HashSet<const PointsToGraph::MemoryNode*> &
AgnosticMemoryNodeProvider::GetCallExitNodes(const CallNode&) const
{
  return MemoryNodes_;
}

HashSet<const PointsToGraph::MemoryNode*>
AgnosticMemoryNodeProvider::GetOutputNodes(const jive::output & output) const
{
  JLM_ASSERT(is<PointerType>(output.type()));
  auto & registerNode = PointsToGraph_.GetRegisterNode(output);

  HashSet<const PointsToGraph::MemoryNode*> memoryNodes;
  for (auto & memoryNode : registerNode.Targets())
    memoryNodes.Insert(&memoryNode);

  return memoryNodes;
}

}