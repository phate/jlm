/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/util/Math.hpp>


namespace jlm::llvm::aa
{

/**
 * Converts a PointerObjectSet into PointsToGraph nodes,
 * and points-to-graph set memberships into edges.
 *
 * Note that registers sharing PointerObject, become separate PointsToGraph nodes.
 *
 * An *escaped* node is not included.
 * Instead implicit edges through escaped+external, are added as explicit edges.
 * @return the newly created PointsToGraph
 */
std::unique_ptr<PointsToGraph>
Andersen::ConstructPointsToGraphFromPointerObjectSet(const PointerObjectSet& set)
{
  auto pointsToGraph = PointsToGraph::Create();

  // memory nodes are the nodes that can be pointed to in the points-to graph.
  // This vector has the same indexing as the nodes themselves, register nodes become nullptr.
  std::vector<PointsToGraph::MemoryNode *> memoryNodes(set.NumPointerObjects());

  // Nodes that should point to external in the final graph.
  // They also get explicit edges connecting them to all escaped memory nodes.
  std::vector<PointsToGraph::Node *> pointsToExternal;

  // A list of all memory nodes that have been marked as escaped
  std::vector<PointsToGraph::MemoryNode *> escapedMemoryNodes;

  // First all memory nodes are created
  for (auto [allocaNode, pointerObjectIndex] : set.GetAllocaMap()) {
    auto& node = PointsToGraph::AllocaNode::Create(*pointsToGraph, *allocaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [mallocNode, pointerObjectIndex] : set.GetMallocMap()) {
    auto& node = PointsToGraph::MallocNode::Create(*pointsToGraph, *mallocNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [deltaNode, pointerObjectIndex] : set.GetGlobalMap()) {
    auto& node = PointsToGraph::DeltaNode::Create(*pointsToGraph, *deltaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [lambdaNode, pointerObjectIndex] : set.GetFunctionMap()) {
    auto& node = PointsToGraph::LambdaNode::Create(*pointsToGraph, *lambdaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [argument, pointerObjectIndex] : set.GetImportMap()) {
    auto& node = PointsToGraph::ImportNode::Create(*pointsToGraph, *argument);
    memoryNodes[pointerObjectIndex] = &node;
  }

  // Helper function for attaching PointsToGraph nodes to their pointees, based on the PointerObject's points-to set.
  auto applyPointsToSet = [&](PointsToGraph::Node & node, PointerObject::Index index)
  {
    // Add all PointsToGraph nodes who should point to external to the list
    if (set.GetPointerObject(index).PointsToExternal())
      pointsToExternal.push_back(&node);

    for (PointerObject::Index targetIdx : set.GetPointsToSet(index)) {
      // Only PointerObjects corresponding to memory nodes can be members of points-to sets
      JLM_ASSERT(memoryNodes[targetIdx]);
      node.AddEdge(*memoryNodes[targetIdx]);
    }
  };

  // Now add register nodes last. While adding them, also add any edges from them to the previously created memoryNodes
  for (auto [outputNode, registerIdx] : set.GetRegisterMap()) {
    auto &registerNode = PointsToGraph::RegisterNode::Create(*pointsToGraph, *outputNode);
    applyPointsToSet(registerNode, registerIdx);
  }

  // Now add all edges from memory node to memory node. Also tracks which memory nodes are marked as escaped
  for (PointerObject::Index idx = 0; idx < set.NumPointerObjects(); idx++) {
    if (memoryNodes[idx] == nullptr)
      continue; // Skip all nodes that are not MemoryNodes

    applyPointsToSet(*memoryNodes[idx], idx);

    if (set.GetPointerObject(idx).HasEscaped())
      escapedMemoryNodes.push_back(memoryNodes[idx]);
  }

  // Finally make all nodes marked as pointing to external, point to all escaped memory nodes in the graph
  for (auto source : pointsToExternal) {
    for (auto target : escapedMemoryNodes) {
      source->AddEdge(*target);
    }
    // Add an edge to the special PointsToGraph node called "external" as well
    source->AddEdge(pointsToGraph->GetExternalMemoryNode());
  }

  return pointsToGraph;
}

std::unique_ptr<PointsToGraph>
Andersen::Analyze(const RvsdgModule &module, jlm::util::StatisticsCollector &statisticsCollector)
{
  Set_ = std::make_unique<PointerObjectSet>();
  Constraints_ = std::make_unique<PointerObjectConstraintSet>(*Set_);

  AnalyzeRvsdg(module.Rvsdg());

  auto result = ConstructPointsToGraphFromPointerObjectSet(*Set_);

  Constraints_.reset();
  Set_.reset();

  return result;
}

}