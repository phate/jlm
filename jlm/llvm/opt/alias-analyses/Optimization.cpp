/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/Optimization.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>

namespace jlm::aa {

/*
 * FIXME: We should resolve the unknown memory node somewhere else. Preferably already in the alias analysis pass.
 */
static void
UnlinkUnknownMemoryNode(PointsToGraph & pointsToGraph)
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
}

SteensgaardAgnostic::~SteensgaardAgnostic() noexcept
= default;

void
SteensgaardAgnostic::run(
  RvsdgModule & rvsdgModule,
  StatisticsCollector & statisticsCollector)
{
  Steensgaard steensgaard;
  auto pointsToGraph = steensgaard.Analyze(rvsdgModule, statisticsCollector);
  UnlinkUnknownMemoryNode(*pointsToGraph);

  auto provisioning = AgnosticMemoryNodeProvider::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
}

SteensgaardRegionAware::~SteensgaardRegionAware() noexcept
= default;

void
SteensgaardRegionAware::run(
  RvsdgModule & rvsdgModule,
  StatisticsCollector & statisticsCollector)
{
  Steensgaard steensgaard;
  auto pointsToGraph = steensgaard.Analyze(rvsdgModule, statisticsCollector);
  UnlinkUnknownMemoryNode(*pointsToGraph);

  auto provisioning = RegionAwareMemoryNodeProvider::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
}

}