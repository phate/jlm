/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/opt/alias-analyses/BasicEncoder.hpp>
#include <jlm/opt/alias-analyses/Optimization.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>

namespace jlm::aa {

SteensgaardBasic::~SteensgaardBasic() noexcept
= default;

void
SteensgaardBasic::run(
  RvsdgModule & rvsdgModule,
  const StatisticsDescriptor & statisticsDescriptor)
{
  auto UnlinkUnknownMemoryNode = [](PointsToGraph & pointsToGraph)
  {
    std::vector<PointsToGraph::Node*> memoryNodes;
    for (auto & allocaNode : pointsToGraph.AllocaNodes())
      memoryNodes.push_back(&allocaNode);

    for (auto & deltaNode : pointsToGraph.DeltaNodes())
      memoryNodes.push_back(&deltaNode);

    for (auto & lambdaNode : pointsToGraph.LambdaNodes())
      memoryNodes.push_back(&lambdaNode);

    for (auto & mallocNode : pointsToGraph.MallocNodes())
      memoryNodes.push_back(&mallocNode);

    for (auto & node : pointsToGraph.ImportNodes())
      memoryNodes.push_back(&node);

    auto & unknownMemoryNode = pointsToGraph.GetUnknownMemoryNode();
    while (unknownMemoryNode.NumSources() != 0) {
      auto & source = *unknownMemoryNode.Sources().begin();
      for (auto & memoryNode : memoryNodes)
        source.AddEdge(*dynamic_cast<PointsToGraph::MemoryNode *>(memoryNode));
      source.RemoveEdge(unknownMemoryNode);
    }
  };

  Steensgaard steensgaard;
  auto pointsToGraph = steensgaard.Analyze(rvsdgModule, statisticsDescriptor);
  UnlinkUnknownMemoryNode(*pointsToGraph);

  BasicEncoder encoder(*pointsToGraph);
  encoder.Encode(rvsdgModule, statisticsDescriptor);
}

}
