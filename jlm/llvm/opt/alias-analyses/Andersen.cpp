/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::llvm::aa
{

void
Andersen::AnalyzeSimpleNode(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeAlloca(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeMalloc(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeLoad(const LoadNode & loadNode)
{

}

void
Andersen::AnalyzeStore(const StoreNode & storeNode)
{

}

void
Andersen::AnalyzeCall(const CallNode & callNode)
{

}

void
Andersen::AnalyzeGep(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeBitcast(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeBits2ptr(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeConstantPointerNull(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeUndef(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeMemcpy(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeConstantArray(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeConstantStruct(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeConstantAggregateZero(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeExtractValue(const rvsdg::simple_node & node)
{

}

void
Andersen::AnalyzeLambda(const lambda::node & node)
{

}

void
Andersen::AnalyzeDelta(const delta::node & node)
{

}

void
Andersen::AnalyzePhi(const phi::node & node)
{
  // Handle context variables
  for (auto cv = node.begin_cv(); cv != node.end_cv(); ++cv) {
    if (!is<PointerType>(cv->type()))
      continue;

    auto & inputRegister = *cv->origin();
    const auto inputRegisterPO = Set_->GetOrCreateRegisterPointerObject(inputRegister);

    Set_->MapRegisterToExistingPointerObject(*cv->argument(), inputRegisterPO);
  }

  // Handle recursion variable arguments
  for (auto rv = node.begin_rv(); rv != node.end_rv(); ++rv) {
    if (!is<PointerType>(rv->type()))
      continue;

    *rv->argument()

  }

  // Handle subregion
  AnalyzeRegion(*node.subregion());

  // Handle recursion variable outputs
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
    if (!is<PointerType>(rv->type()))
      continue;

    auto & origin = LocationSet_->Find(*rv->result()->origin());
    auto & argument = LocationSet_->Find(*rv->argument());
    LocationSet_->Join(origin, argument);

    auto & output = LocationSet_->FindOrInsertRegisterLocation(
      *rv.output(),
      PointsToFlags::PointsToNone);
    LocationSet_->Join(argument, output);
  }
}

void
Andersen::AnalyzeGamma(const rvsdg::gamma_node & node)
{
  // Handle input variables
  for (auto ev = node.begin_entryvar(); ev != node.end_entryvar(); ++ev)
  {
    if (!jlm::rvsdg::is<PointerType>(ev->type()))
      continue;

    auto & inputRegister = *ev->origin();
    const auto inputRegisterPO = Set_->GetOrCreateRegisterPointerObject(inputRegister);

    for (auto & argument : *ev)
      Set_->MapRegisterToExistingPointerObject(argument, inputRegisterPO);
  }

  // Handle subregions
  for (size_t n = 0; n < node.nsubregions(); n++)
    AnalyzeRegion(*node.subregion(n));

  // Handle exit variables
  for (auto ex = node.begin_exitvar(); ex != node.end_exitvar(); ++ex)
  {
    if (!jlm::rvsdg::is<PointerType>(ex->type()))
      continue;

    auto & outputRegister = *ex.output();
    const auto outputRegisterPO = Set_->GetOrCreateRegisterPointerObject(outputRegister);

    for (auto & result : *ex) {
      const auto resultRegisterPO = Set_->GetRegisterPointerObject(*result.origin());
      Constraints_->AddConstraint(SupersetConstraint(outputRegisterPO, resultRegisterPO));
    }
  }
}

void
Andersen::AnalyzeTheta(const rvsdg::theta_node & node)
{
  AnalyzeRegion(*node.subregion());

  // Make each loop variable a superset of its entry and the inner region's result
  for (const rvsdg::theta_output* thetaOutput : node) {
    if (!jlm::rvsdg::is<PointerType>(thetaOutput->type()))
      continue;

    auto & inputRegister = *thetaOutput->input()->origin();
    auto & innerResultRegister = *thetaOutput->result()->origin();
    auto & innerArgumentRegister = *thetaOutput->argument();

    const auto inputRegisterPO = Set_->GetOrCreateRegisterPointerObject(inputRegister);
    const auto innerResultRegisterPO = Set_->GetOrCreateRegisterPointerObject(innerResultRegister);
    const auto innerArgumentRegisterPO = Set_->GetOrCreateRegisterPointerObject(innerArgumentRegister);

    Constraints_->AddConstraint(SupersetConstraint(innerArgumentRegisterPO, inputRegisterPO));
    Constraints_->AddConstraint(SupersetConstraint(innerArgumentRegisterPO, innerResultRegisterPO));

    // Finally, the output of the theta node itself has the same points-to-set as the inner argument
    Set_->MapRegisterToExistingPointerObject(*thetaOutput, innerArgumentRegisterPO);
  }
}

void
Andersen::AnalyzeStructuralNode(const rvsdg::structural_node & node)
{
  switch(typeid(node.operation()))
  {
  case typeid(lambda::operation):
    return AnalyzeLambda(*util::AssertedCast<const lambda::node>(&node));
  case typeid(delta::operation):
    return AnalyzeDelta(*util::AssertedCast<const delta::node>(&node));
  case typeid(phi::operation):
    return AnalyzePhi(*util::AssertedCast<const phi::node>(&node));
  case typeid(rvsdg::gamma_op):
    return AnalyzeGamma(*util::AssertedCast<const rvsdg::gamma_node>(&node));
  case typeid(rvsdg::theta_op):
    return AnalyzeTheta(*util::AssertedCast<const rvsdg::theta_node>(&node));
  default:
    JLM_UNREACHABLE("Unknown structural node operation");
  }
}

void
Andersen::AnalyzeRegion(rvsdg::region & region)
{
  rvsdg::topdown_traverser traverser(&region);
  for (const auto node : traverser)
  {
    if (auto simpleNode = dynamic_cast<const rvsdg::simple_node*>(node))
      AnalyzeSimpleNode(*simpleNode);
    else if (auto structuralNode = dynamic_cast<const rvsdg::structural_node*>(node))
      AnalyzeStructuralNode(*structuralNode);
    else
      JLM_UNREACHABLE("Unknown node type");
  }
}

void
Andersen::AnalyzeRvsdg(const rvsdg::graph & graph)
{
  auto & rootRegion = *graph.root();

  // Iterate over all arguments to the root region - symbols imported from other modules
  for (size_t n = 0; n < rootRegion.narguments(); n++) {
    auto & argument = *rootRegion.argument(n);

    // Only care about imported pointer values
    if (!jlm::rvsdg::is<PointerType>(argument.type()))
      continue;

    /* FIXME: Steensgaard says we should not add function imports.
     * What should be done to functions instead, creating lambda nodes with special linkage flags? */

    // Create a memory PointerObject representing the target of the external symbol
    // We can assume that two external symbols don't alias, clang does.
    const PointerObject::Index importObject = Set_->CreateImportMemoryObject(argument);

    // Create a register PointerObject representing the address value itself
    const PointerObject::Index importRegister = Set_->CreateRegisterPointerObject(argument);
    Constraints_->AddPointerPointeeConstraint(importRegister, importObject);
  }

  AnalyzeRegion(rootRegion);

  // Mark all results escaping the root module as escaped
  for (size_t n = 0; n < rootRegion.nresults(); n++) {
    auto& result = *rootRegion.result(n);
    JLM_ASSERT(result.origin());
    const PointerObject::Index escapedRegister = Set_->GetRegisterMap().at(result.origin());
    Constraints_->AddRegisterContentEscapedConstraint(escapedRegister);
  }
}

std::unique_ptr<PointsToGraph>
Andersen::Analyze(const RvsdgModule &module, util::StatisticsCollector &statisticsCollector)
{
  Set_ = std::make_unique<PointerObjectSet>();
  Constraints_ = std::make_unique<PointerObjectConstraintSet>(*Set_);

  AnalyzeRvsdg(module.Rvsdg());

  auto result = ConstructPointsToGraphFromPointerObjectSet(*Set_);

  Constraints_.reset();
  Set_.reset();

  return result;
}

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

    for (const auto targetIdx : set.GetPointsToSet(index)) {
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
  for (const auto source : pointsToExternal) {
    for (const auto target : escapedMemoryNodes) {
      source->AddEdge(*target);
    }
    // Add an edge to the special PointsToGraph node called "external" as well
    source->AddEdge(pointsToGraph->GetExternalMemoryNode());
  }

  return pointsToGraph;
}

}
