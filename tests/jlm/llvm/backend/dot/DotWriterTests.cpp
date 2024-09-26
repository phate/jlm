/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/backend/dot/DotWriter.hpp>
#include <jlm/llvm/ir/operators.hpp>

#include <cassert>

static int
TestWriteGraphs()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  jlm::tests::GammaTest gammaTest;

  // Act
  GraphWriter writer;
  dot::WriteGraphs(writer, *gammaTest.graph().root(), false);

  writer.OutputAllGraphs(std::cout, GraphOutputFormat::Dot);

  // Assert
  auto & rootGraph = writer.GetGraph(0);
  assert(rootGraph.GetProgramObject() == reinterpret_cast<uintptr_t>(gammaTest.graph().root()));
  assert(rootGraph.NumNodes() == 1);       // Only the lambda node for "f"
  assert(rootGraph.NumResultNodes() == 1); // Exporting the function "f"
  auto & lambdaNode = *AssertedCast<InOutNode>(&rootGraph.GetNode(0));

  // The lambda only has one output, and a single subgraph
  assert(lambdaNode.GetLabel() == gammaTest.lambda->operation().debug_string());
  assert(lambdaNode.NumInputPorts() == 0);
  assert(lambdaNode.NumOutputPorts() == 1);
  assert(lambdaNode.NumSubgraphs() == 1);

  auto & fctBody = lambdaNode.GetSubgraph(0);
  assert(fctBody.NumArgumentNodes() == 6);
  assert(fctBody.NumResultNodes() == 2);

  // Argument a1 leads to the gamma node
  auto & connections = fctBody.GetArgumentNode(1).GetConnections();
  assert(connections.size() == 1);
  auto & gammaNode = *AssertedCast<InOutNode>(&connections[0]->GetTo().GetNode());
  assert(gammaNode.GetLabel() == gammaTest.gamma->operation().debug_string());
  assert(gammaNode.NumInputPorts() == 5);
  assert(gammaNode.NumOutputPorts() == 2);
  assert(gammaNode.NumSubgraphs() == 2);

  // The first argument of the first region of the gamma references the second gamma input
  auto & argument = gammaNode.GetSubgraph(0).GetArgumentNode(0);
  auto & input = gammaNode.GetInputPort(1);
  assert(argument.GetAttributeGraphElement("input") == &input);
  // The label also includes the attribute index and input index
  assert(argument.GetLabel() == "a0 <- i1");
  auto & result = argument.GetConnections().front()->GetOtherEnd(argument);
  assert(result.GetLabel() == "r0 -> o0");

  // Check that the last argument is colored red to represent the memory state type
  auto & stateConnections = fctBody.GetArgumentNode(5).GetConnections();
  assert(stateConnections.size() == 1);
  assert(stateConnections.front()->GetAttributeString("color") == "#FF0000");

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/dot/DotWriterTests-TestWriteGraphs", TestWriteGraphs)

static int
TestTypeGraph()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  jlm::tests::GammaTest gammaTest;
  auto ptrType = PointerType::Create();
  auto bit32Type = jlm::rvsdg::bittype::Create(32);
  auto memType = MemoryStateType::Create();

  // Act
  GraphWriter writer;
  dot::WriteGraphs(writer, *gammaTest.graph().root(), true);

  writer.Finalize();
  writer.OutputAllGraphs(std::cout, GraphOutputFormat::Dot);

  // Assert
  auto & typeGraph = writer.GetGraph(0);
  assert(typeGraph.GetProgramObject() == 0);

  // Check that nodes exist for the given types
  [[maybe_unused]] auto & ptrNode = typeGraph.GetFromProgramObject<Node>(*ptrType);
  [[maybe_unused]] auto & bit32Node = typeGraph.GetFromProgramObject<Node>(*ptrType);
  auto & memNode = typeGraph.GetFromProgramObject<Node>(*memType);

  // Check that the rightmost argument of the function references the memNode type
  auto & fGraph = writer.GetGraph(2);
  assert(writer.GetElementFromProgramObject(*gammaTest.lambda->subregion()) == &fGraph);
  assert(fGraph.GetArgumentNode(5).GetAttributeGraphElement("type") == &memNode);

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/dot/DotWriterTests-TestTypeGraph", TestTypeGraph)
