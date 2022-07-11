/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/opt/alias-analyses/BasicMemoryNodeProvider.hpp>

namespace jlm::aa
{

BasicMemoryNodeProvider::BasicMemoryNodeProvider(const PointsToGraph & pointsToGraph)
  : PointsToGraph_(pointsToGraph)
{
  CollectMemoryNodes(pointsToGraph);
}

const PointsToGraph &
BasicMemoryNodeProvider::GetPointsToGraph() const noexcept
{
  return PointsToGraph_;
}

const std::vector<const PointsToGraph::MemoryNode*> &
BasicMemoryNodeProvider::GetRegionEntryNodes(const jive::region&) const
{
  return MemoryNodes_;
}

const std::vector<const PointsToGraph::MemoryNode*> &
BasicMemoryNodeProvider::GetRegionExitNodes(const jive::region&) const
{
  return MemoryNodes_;
}

const std::vector<const PointsToGraph::MemoryNode*> &
BasicMemoryNodeProvider::GetCallEntryNodes(const CallNode&) const
{
  return MemoryNodes_;
}

const std::vector<const PointsToGraph::MemoryNode*> &
BasicMemoryNodeProvider::GetCallExitNodes(const CallNode&) const
{
  return MemoryNodes_;
}

std::vector<const PointsToGraph::MemoryNode*>
BasicMemoryNodeProvider::GetOutputNodes(const jive::output & output) const
{
  JLM_ASSERT(is<PointerType>(output.type()));
  auto & registerNode = PointsToGraph_.GetRegisterNode(output);

  std::vector<const PointsToGraph::MemoryNode*> memoryNodes;
  for (auto & memoryNode : registerNode.Targets())
    memoryNodes.push_back(&memoryNode);

  return memoryNodes;
}

void
BasicMemoryNodeProvider::CollectMemoryNodes(const PointsToGraph & pointsToGraph)
{
  for (auto & allocaNode : pointsToGraph.AllocaNodes())
    MemoryNodes_.push_back(&allocaNode);

  for (auto & deltaNode : pointsToGraph.DeltaNodes())
    MemoryNodes_.push_back(&deltaNode);

  for (auto & lambdaNode : pointsToGraph.LambdaNodes())
    MemoryNodes_.push_back(&lambdaNode);

  for (auto & mallocNode : pointsToGraph.MallocNodes())
    MemoryNodes_.push_back(&mallocNode);

  for (auto & importNode : pointsToGraph.ImportNodes())
    MemoryNodes_.push_back(&importNode);

  MemoryNodes_.push_back(&pointsToGraph.GetExternalMemoryNode());
}

}