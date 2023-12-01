/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <typeinfo>

namespace jlm::llvm::aa
{

void
Andersen::AnalyzeSimpleNode(const rvsdg::simple_node & node)
{
  const auto & op = node.operation();
  const auto & type = typeid(op);

  if (type == typeid(alloca_op))
    AnalyzeAlloca(node);
  else if (type == typeid(malloc_op))
    AnalyzeMalloc(node);
  else if (type == typeid(LoadOperation))
    AnalyzeLoad(*util::AssertedCast<const LoadNode>(&node));
  else if (type == typeid(StoreOperation))
    AnalyzeStore(*util::AssertedCast<const StoreNode>(&node));
  else if (type == typeid(CallOperation))
    AnalyzeCall(*util::AssertedCast<const CallNode>(&node));
  else if (type == typeid(GetElementPtrOperation))
    AnalyzeGep(node);
  else if (type == typeid(bitcast_op))
    AnalyzeBitcast(node);
  else if (type == typeid(bits2ptr_op))
    AnalyzeBits2ptr(node);
  else if (type == typeid(ConstantPointerNullOperation))
    AnalyzeConstantPointerNull(node);
  else if (type == typeid(UndefValueOperation))
    AnalyzeUndef(node);
  else if (type == typeid(Memcpy))
    AnalyzeMemcpy(node);
  else if (type == typeid(ConstantArray))
    AnalyzeConstantArray(node);
  else if (type == typeid(ConstantStruct))
    AnalyzeConstantStruct(node);
  else if (type == typeid(ConstantAggregateZero))
    AnalyzeConstantAggregateZero(node);
  else if (type == typeid(ExtractValue))
    AnalyzeExtractValue(node);
}

void
Andersen::AnalyzeAlloca(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<alloca_op>(&node));

  const auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
  const auto allocaPO = Set_->CreateAllocaMemoryObject(node);
  Constraints_->AddPointerPointeeConstraint(outputRegisterPO, allocaPO);
}

void
Andersen::AnalyzeMalloc(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<malloc_op>(&node));

  const auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
  const auto mallocPO = Set_->CreateMallocMemoryObject(node);
  Constraints_->AddPointerPointeeConstraint(outputRegisterPO, mallocPO);
}

void
Andersen::AnalyzeLoad(const LoadNode & loadNode)
{
  const auto & addressRegister = *loadNode.GetAddressInput()->origin();
  const auto & outputRegister = *loadNode.GetValueOutput();

  const auto addressRegisterPO = Set_->GetRegisterPointerObject(addressRegister);

  if (!is<PointerType>(outputRegister.type()))
  {
    // TODO: When reading address as an integer, some of address' target might still pointers,
    // which should now be considered as having escaped
    return;
  }

  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
  Constraints_->AddConstraint(SupersetOfAllPointeesConstraint(outputRegisterPO, addressRegisterPO));
}

void
Andersen::AnalyzeStore(const StoreNode & storeNode)
{
  const auto & addressRegister = *storeNode.GetAddressInput()->origin();
  const auto & valueRegister = *storeNode.GetValueInput()->origin();

  // If the written value is not a pointer, be conservative and make the address
  if (!is<PointerType>(valueRegister.type()))
  {
    // TODO: We are writing an integer to *address,
    // which really should mark all of address' targets as pointing to external
    // in case they are ever read as pointers.
    return;
  }

  const auto addressRegisterPO = Set_->GetRegisterPointerObject(addressRegister);
  const auto valueRegisterPO = Set_->GetRegisterPointerObject(valueRegister);
  Constraints_->AddConstraint(
      AllPointeesPointToSupersetConstraint(addressRegisterPO, valueRegisterPO));
}

void
Andersen::AnalyzeCall(const CallNode & callNode)
{
  // The address being called by the call node
  const auto & callTarget = *callNode.GetFunctionInput()->origin();
  const auto callTargetPO = Set_->GetRegisterPointerObject(callTarget);

  // Create PointerObjects for all output values of pointer type
  for (size_t n = 0; n < callNode.NumResults(); n++)
  {
    const auto & outputRegister = *callNode.Result(n);
    Set_->CreateRegisterPointerObject(outputRegister);
  }

  // We make no attempt at detecting what type of call this is here.
  // The logic handling external and indirect calls is done by the FunctionCallConstraint.
  // Passing points-to-sets from call-site to function bodies is done fully by this constraint.
  Constraints_->AddConstraint(FunctionCallConstraint(callTargetPO, callNode));
}

void
Andersen::AnalyzeGep(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<GetElementPtrOperation>(&node));

  // The analysis is field insensitive, so ignoring the offset and mapping the output
  // to the same PointerObject as the input is sufficient.
  const auto & baseRegister = *node.input(0)->origin();
  const auto & outputRegister = *node.output(0);

  const auto baseRegisterPO = Set_->GetRegisterPointerObject(baseRegister);
  Set_->MapRegisterToExistingPointerObject(outputRegister, baseRegisterPO);
}

void
Andersen::AnalyzeBitcast(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<bitcast_op>(&node));

  const auto & inputRegister = *node.input(0)->origin();
  const auto & outputRegister = *node.output(0);

  // TODO: bitcast can never work with aggregate types, but it can be a vector of pointers
  if (!is<PointerType>(inputRegister.type()))
    return;

  // If the input is a pointer type, the output must also be a pointer type
  JLM_ASSERT(is<PointerType>(outputRegister.type()));

  const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
  Set_->MapRegisterToExistingPointerObject(outputRegister, inputRegisterPO);
}

void
Andersen::AnalyzeBits2ptr(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<bits2ptr_op>(&node));
  const auto & output = *node.output(0);
  JLM_ASSERT(is<PointerType>(output.type()));

  // This operation synthesizes a pointer from bytes.
  // Since no points-to information is tracked through integers, the resulting pointer must
  // be assumed to possibly point to any external or escaped memory object.
  const auto outputPO = Set_->CreateRegisterPointerObject(output);
  Constraints_->AddPointsToExternalConstraint(outputPO);
}

void
Andersen::AnalyzeConstantPointerNull(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantPointerNullOperation>(&node));
  const auto & output = *node.output(0);
  JLM_ASSERT(is<PointerType>(output.type()));

  // ConstantPointerNull cannot point to any memory location. We therefore only insert a register
  // node for it, but let this node not point to anything.
  Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeUndef(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<UndefValueOperation>(&node));
  const auto & output = *node.output(0);

  if (!is<PointerType>(output.type()))
    return;

  // UndefValue cannot point to any memory location. We therefore only insert a register node for
  // it, but let this node not point to anything.
  Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeMemcpy(const rvsdg::simple_node & node)
{
  auto & dstAddressRegister = *node.input(0)->origin();
  auto & srcAddressRegister = *node.input(1)->origin();
  JLM_ASSERT(is<PointerType>(dstAddressRegister.type()));
  JLM_ASSERT(is<PointerType>(srcAddressRegister.type()));

  const auto dstAddressRegisterPO = Set_->GetRegisterPointerObject(dstAddressRegister);
  const auto srcAddressRegisterPO = Set_->GetRegisterPointerObject(srcAddressRegister);

  // Create an intermediate PointerObject representing the moved values
  const auto dummyPO = Set_->CreateDummyRegisterPointerObject();

  // Add a "load" constraint from the source into the dummy register
  Constraints_->AddConstraint(SupersetOfAllPointeesConstraint(dummyPO, srcAddressRegisterPO));
  // Add a "store" constraint from the dummy register into the destination
  Constraints_->AddConstraint(AllPointeesPointToSupersetConstraint(dstAddressRegisterPO, dummyPO));
}

void
Andersen::AnalyzeConstantArray(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantStruct>(&node));

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    const auto & inputRegister = *node.input(n)->origin();
    if (!is<PointerType>(inputRegister.type()))
      continue;

    // TODO: Pass pointer information through aggregate types
    // Since the rest of the code only considers values of PointerType, mark inputs as escaping.
    auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Constraints_->AddRegisterContentEscapedConstraint(inputRegisterPO);
  }
}

void
Andersen::AnalyzeConstantStruct(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantStruct>(&node));

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    const auto & inputRegister = *node.input(n)->origin();
    if (!is<PointerType>(inputRegister.type()))
      continue;

    // TODO: Pass pointer information through aggregate types
    // Since the rest of the code only considers values of PointerType, mark inputs as escaping.
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Constraints_->AddRegisterContentEscapedConstraint(inputRegisterPO);
  }
}

void
Andersen::AnalyzeConstantAggregateZero(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantAggregateZero>(&node));
  auto & output = *node.output(0);

  if (!is<PointerType>(output.type()))
    return;

  // ConstantAggregateZero cannot point to any memory location.
  // We therefore only insert a register node for it, but let this node not point to anything.
  Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeExtractValue(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ExtractValue>(&node));

  const auto & result = *node.output(0);
  if (!is<PointerType>(result.type()))
    return;

  // TODO: Make aggregate types with at least one field of pointer type have PointerObjects.
  // Then we could be more precise than "escaping" all pointers passing through aggregate types.
  // This involves replacing all usages of is<PointerType> with IsOrContains<PointerType>
  const auto resultPO = Set_->CreateRegisterPointerObject(result);
  Constraints_->AddPointsToExternalConstraint(resultPO);
}

void
Andersen::AnalyzeStructuralNode(const rvsdg::structural_node & node)
{
  const auto & op = node.operation();
  const auto & type = typeid(op);

  if (type == typeid(lambda::operation))
    AnalyzeLambda(*util::AssertedCast<const lambda::node>(&node));
  else if (type == typeid(delta::operation))
    AnalyzeDelta(*util::AssertedCast<const delta::node>(&node));
  else if (type == typeid(phi::operation))
    AnalyzePhi(*util::AssertedCast<const phi::node>(&node));
  else if (type == typeid(rvsdg::gamma_op))
    AnalyzeGamma(*util::AssertedCast<const rvsdg::gamma_node>(&node));
  else if (type == typeid(rvsdg::theta_op))
    AnalyzeTheta(*util::AssertedCast<const rvsdg::theta_node>(&node));
  else
    JLM_UNREACHABLE("Unknown structural node operation");
}

void
Andersen::AnalyzeLambda(const lambda::node & lambda)
{
  // Handle context variables
  for (auto & cv : lambda.ctxvars())
  {
    if (!jlm::rvsdg::is<PointerType>(cv.type()))
      continue;

    auto & inputRegister = *cv.origin();
    auto & argumentRegister = *cv.argument();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  // Create Register PointerObjects for each argument in the function
  for (auto & argument : lambda.fctarguments())
  {
    if (jlm::rvsdg::is<PointerType>(argument.type()))
      Set_->CreateRegisterPointerObject(argument);
  }

  // Analyze the subregion
  AnalyzeRegion(*lambda.subregion());

  // Create a lambda PointerObject for the lambda itself
  const auto lambdaPO = Set_->CreateFunctionMemoryObject(lambda);

  // Make the labda node's output point to the lambda PointerObject
  const auto & lambdaOutput = *lambda.output();
  const auto lambdaOutputPO = Set_->CreateRegisterPointerObject(lambdaOutput);
  Constraints_->AddPointerPointeeConstraint(lambdaOutputPO, lambdaPO);

  // If the function escapes the module, all arguments must point to external,
  // and the return value must be marked as escaping the module.
  Constraints_->AddConstraint(HandleEscapingFunctionConstraint(lambdaPO));
}

void
Andersen::AnalyzeDelta(const delta::node & delta)
{
  // Hanfle context variables
  for (auto & cv : delta.ctxvars())
  {
    if (!is<PointerType>(cv.type()))
      continue;

    auto & inputRegister = *cv.origin();
    auto & argumentRegister = *cv.argument();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  // Analyze the internal region
  AnalyzeRegion(*delta.subregion());

  // Map the result register PointerObject to the output register
  auto & resultRegister = *delta.result()->origin();
  auto & outputRegister = *delta.output();
  const auto resultRegisterPO = Set_->GetRegisterPointerObject(resultRegister);
  Set_->MapRegisterToExistingPointerObject(outputRegister, resultRegisterPO);
}

void
Andersen::AnalyzePhi(const phi::node & phi)
{
  // Handle context variables
  for (auto cv = phi.begin_cv(); cv != phi.end_cv(); ++cv)
  {
    if (!is<PointerType>(cv->type()))
      continue;

    auto & inputRegister = *cv->origin();
    auto & argumentRegister = *cv->argument();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  // Create Register PointerObjects for each recursion variable argument
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); ++rv)
  {
    if (!is<PointerType>(rv->type()))
      continue;

    auto & argumentRegister = *rv->argument();
    Set_->CreateRegisterPointerObject(argumentRegister);
  }

  // Handle subregion
  AnalyzeRegion(*phi.subregion());

  // Handle recursion variable results
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); ++rv)
  {
    if (!is<PointerType>(rv->type()))
      continue;

    // Make the recursion variable argument point to what the result register points to
    auto & argumentRegister = *rv->argument();
    auto & resultRegister = *rv->result()->origin();
    const auto argumentRegisterPO = Set_->GetRegisterPointerObject(argumentRegister);
    const auto resultRegisterPO = Set_->GetRegisterPointerObject(resultRegister);
    Constraints_->AddConstraint(SupersetConstraint(argumentRegisterPO, resultRegisterPO));

    // Map the output register to the recursion result's pointer object
    auto & outputRegister = *rv;
    Set_->MapRegisterToExistingPointerObject(outputRegister, argumentRegisterPO);
  }
}

void
Andersen::AnalyzeGamma(const rvsdg::gamma_node & gamma)
{
  // Handle input variables
  for (auto ev = gamma.begin_entryvar(); ev != gamma.end_entryvar(); ++ev)
  {
    if (!is<PointerType>(ev->type()))
      continue;

    auto & inputRegister = *ev->origin();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);

    for (auto & argument : *ev)
      Set_->MapRegisterToExistingPointerObject(argument, inputRegisterPO);
  }

  // Handle subregions
  for (size_t n = 0; n < gamma.nsubregions(); n++)
    AnalyzeRegion(*gamma.subregion(n));

  // Handle exit variables
  for (auto ex = gamma.begin_exitvar(); ex != gamma.end_exitvar(); ++ex)
  {
    if (!is<PointerType>(ex->type()))
      continue;

    auto & outputRegister = *ex.output();
    const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);

    for (auto & result : *ex)
    {
      const auto resultRegisterPO = Set_->GetRegisterPointerObject(*result.origin());
      Constraints_->AddConstraint(SupersetConstraint(outputRegisterPO, resultRegisterPO));
    }
  }
}

void
Andersen::AnalyzeTheta(const rvsdg::theta_node & theta)
{
  // Create a PointerObject for each argument in the inner region
  // And make it point to a superset of the corresponding input register
  for (const rvsdg::theta_output * thetaOutput : theta)
  {
    if (!is<PointerType>(thetaOutput->type()))
      continue;

    auto & inputReg = *thetaOutput->input()->origin();
    auto & innerArgumentReg = *thetaOutput->argument();
    const auto inputRegPO = Set_->GetRegisterPointerObject(inputReg);
    const auto innerArgumentRegPO = Set_->CreateRegisterPointerObject(innerArgumentReg);

    // The inner argument can point to anything the input did
    Constraints_->AddConstraint(SupersetConstraint(innerArgumentRegPO, inputRegPO));
  }

  AnalyzeRegion(*theta.subregion());

  // Iterate over loop variables again, making the inner arguments point to a superset
  // of what the corresponding result registers point to
  for (const rvsdg::theta_output * thetaOutput : theta)
  {
    if (!is<PointerType>(thetaOutput->type()))
      continue;

    auto & innerArgumentReg = *thetaOutput->argument();
    auto & innerResultReg = *thetaOutput->result()->origin();
    auto & outputReg = *thetaOutput;

    const auto innerArgumentRegPO = Set_->GetRegisterPointerObject(innerArgumentReg);
    const auto innerResultRegPO = Set_->GetRegisterPointerObject(innerResultReg);

    // The inner argument can point to anything the result of last iteration did
    Constraints_->AddConstraint(SupersetConstraint(innerArgumentRegPO, innerResultRegPO));

    // Due to theta nodes running at least once, the output always comes from the inner results
    Set_->MapRegisterToExistingPointerObject(outputReg, innerResultRegPO);
  }
}

void
Andersen::AnalyzeRegion(rvsdg::region & region)
{
  // The use of the top-down traverser is vital, as it ensures all input origins
  // of pointer type are mapped to PointerObjects by the time a node is processed.
  rvsdg::topdown_traverser traverser(&region);

  // While visiting the node we have the responsibility of creating
  // PointerObjects for any of the node's outputs of pointer type
  for (const auto node : traverser)
  {
    if (auto simpleNode = dynamic_cast<const rvsdg::simple_node *>(node))
      AnalyzeSimpleNode(*simpleNode);
    else if (auto structuralNode = dynamic_cast<const rvsdg::structural_node *>(node))
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
  // These symbols can either be global variables or functions
  for (size_t n = 0; n < rootRegion.narguments(); n++)
  {
    auto & argument = *rootRegion.argument(n);

    // Only care about imported pointer values
    if (!jlm::rvsdg::is<PointerType>(argument.type()))
      continue;

    // Create a memory PointerObject representing the target of the external symbol
    // We can assume that two external symbols don't alias, clang does.
    const PointerObject::Index importObject = Set_->CreateImportMemoryObject(argument);

    // Create a register PointerObject representing the address value itself
    const PointerObject::Index importRegister = Set_->CreateRegisterPointerObject(argument);
    Constraints_->AddPointerPointeeConstraint(importRegister, importObject);
  }

  AnalyzeRegion(rootRegion);

  // Mark all results escaping the root module as escaped
  for (size_t n = 0; n < rootRegion.nresults(); n++)
  {
    auto & escapedRegister = *rootRegion.result(n)->origin();
    if (!jlm::rvsdg::is<PointerType>(escapedRegister.type()))
      continue;

    const auto escapedRegisterPO = Set_->GetRegisterPointerObject(escapedRegister);
    Constraints_->AddRegisterContentEscapedConstraint(escapedRegisterPO);
  }
}

std::unique_ptr<PointsToGraph>
Andersen::Analyze(const RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
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
Andersen::ConstructPointsToGraphFromPointerObjectSet(const PointerObjectSet & set)
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
  for (auto [allocaNode, pointerObjectIndex] : set.GetAllocaMap())
  {
    auto & node = PointsToGraph::AllocaNode::Create(*pointsToGraph, *allocaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [mallocNode, pointerObjectIndex] : set.GetMallocMap())
  {
    auto & node = PointsToGraph::MallocNode::Create(*pointsToGraph, *mallocNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [deltaNode, pointerObjectIndex] : set.GetGlobalMap())
  {
    auto & node = PointsToGraph::DeltaNode::Create(*pointsToGraph, *deltaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [lambdaNode, pointerObjectIndex] : set.GetFunctionMap())
  {
    auto & node = PointsToGraph::LambdaNode::Create(*pointsToGraph, *lambdaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [argument, pointerObjectIndex] : set.GetImportMap())
  {
    auto & node = PointsToGraph::ImportNode::Create(*pointsToGraph, *argument);
    memoryNodes[pointerObjectIndex] = &node;
  }

  // Helper function for attaching PointsToGraph nodes to their pointees, based on the
  // PointerObject's points-to set.
  auto applyPointsToSet = [&](PointsToGraph::Node & node, PointerObject::Index index)
  {
    // Add all PointsToGraph nodes who should point to external to the list
    if (set.GetPointerObject(index).PointsToExternal())
      pointsToExternal.push_back(&node);

    for (const auto targetIdx : set.GetPointsToSet(index).Items())
    {
      // Only PointerObjects corresponding to memory nodes can be members of points-to sets
      JLM_ASSERT(memoryNodes[targetIdx]);
      node.AddEdge(*memoryNodes[targetIdx]);
    }
  };

  // Now add register nodes last. While adding them, also add any edges from them to the previously
  // created memoryNodes
  for (auto [outputNode, registerIdx] : set.GetRegisterMap())
  {
    auto & registerNode = PointsToGraph::RegisterNode::Create(*pointsToGraph, *outputNode);
    applyPointsToSet(registerNode, registerIdx);
  }

  // Now add all edges from memory node to memory node. Also tracks which memory nodes are marked as
  // escaped
  for (PointerObject::Index idx = 0; idx < set.NumPointerObjects(); idx++)
  {
    if (memoryNodes[idx] == nullptr)
      continue; // Skip all nodes that are not MemoryNodes

    applyPointsToSet(*memoryNodes[idx], idx);

    if (set.GetPointerObject(idx).HasEscaped())
      escapedMemoryNodes.push_back(memoryNodes[idx]);
  }

  // Finally make all nodes marked as pointing to external, point to all escaped memory nodes in the
  // graph
  for (const auto source : pointsToExternal)
  {
    for (const auto target : escapedMemoryNodes)
    {
      source->AddEdge(*target);
    }
    // Add an edge to the special PointsToGraph node called "external" as well
    source->AddEdge(pointsToGraph->GetExternalMemoryNode());
  }

  return pointsToGraph;
}
}
