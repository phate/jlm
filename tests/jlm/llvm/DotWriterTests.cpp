/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/DotWriter.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/gamma.hpp>

TEST(DotWriterTests, TestWriteGraphs)
{
  using namespace jlm::llvm;
  using namespace jlm::util;
  using namespace jlm::util::graph;

  // Arrange
  GammaTest gammaTest;

  // Act
  Writer writer;
  LlvmDotWriter dotWriter;
  dotWriter.WriteGraphs(writer, gammaTest.graph().GetRootRegion(), false);

  writer.outputAllGraphs(std::cout, OutputFormat::Dot);

  // Assert
  auto & rootGraph = writer.GetGraph(0);
  EXPECT_EQ(
      rootGraph.GetProgramObject(),
      reinterpret_cast<uintptr_t>(&gammaTest.graph().GetRootRegion()));
  EXPECT_EQ(rootGraph.NumNodes(), 1u);       // Only the lambda node for "f"
  EXPECT_EQ(rootGraph.NumResultNodes(), 1u); // Exporting the function "f"
  auto & lambdaNode = *assertedCast<InOutNode>(&rootGraph.GetNode(0));

  // The lambda only has one output, and a single subgraph
  EXPECT_EQ(lambdaNode.GetLabel(), gammaTest.lambda->DebugString());
  EXPECT_EQ(lambdaNode.NumInputPorts(), 0u);
  EXPECT_EQ(lambdaNode.NumOutputPorts(), 1u);
  EXPECT_EQ(lambdaNode.NumSubgraphs(), 1u);

  auto & fctBody = lambdaNode.GetSubgraph(0);
  EXPECT_EQ(fctBody.NumArgumentNodes(), 6u);
  EXPECT_EQ(fctBody.NumResultNodes(), 2u);

  // Argument a1 leads to the gamma node
  auto & connections = fctBody.GetArgumentNode(1).GetConnections();
  EXPECT_EQ(connections.size(), 1u);
  auto & gammaNode = *assertedCast<InOutNode>(&connections[0]->GetTo().GetNode());
  EXPECT_EQ(gammaNode.GetLabel(), gammaTest.gamma->DebugString());
  EXPECT_EQ(gammaNode.NumInputPorts(), 5u);
  EXPECT_EQ(gammaNode.NumOutputPorts(), 2u);
  EXPECT_EQ(gammaNode.NumSubgraphs(), 2u);

  // The second argument of the first region of the gamma references the second gamma input
  auto & argument = gammaNode.GetSubgraph(0).GetArgumentNode(1);
  auto & input = gammaNode.GetInputPort(1);
  EXPECT_EQ(argument.GetAttributeGraphElement("input"), &input);
  // The label also includes the attribute index and input index
  EXPECT_EQ(argument.GetLabel(), "a1 <- i1");
  auto & result = argument.GetConnections().front()->GetOtherEnd(argument);
  EXPECT_EQ(result.GetLabel(), "r0 -> o0");

  // Check that the last argument is colored red to represent the memory state type
  auto & stateConnections = fctBody.GetArgumentNode(5).GetConnections();
  EXPECT_EQ(stateConnections.size(), 1u);
  EXPECT_EQ(stateConnections.front()->GetAttributeString("color"), "#FF0000");

  // Check that the output of the lambda leads to a graph export
  auto & lambdaConnections = lambdaNode.GetOutputPort(0).GetConnections();
  EXPECT_EQ(lambdaConnections.size(), 1u);
  auto & graphExport = lambdaConnections.front()->GetTo().GetNode();
  EXPECT_EQ(graphExport.GetLabel(), "export[f]");
}

TEST(DotWriterTests, TestWriteGraph)
{
  using namespace jlm::llvm;
  using namespace jlm::util;
  using namespace jlm::util::graph;

  // Arrange
  GammaTest gammaTest;

  // Act
  Writer writer;
  LlvmDotWriter dotWriter;
  dotWriter.WriteGraph(writer, gammaTest.graph().GetRootRegion());

  // Assert
  auto & rootGraph = writer.GetGraph(0);
  EXPECT_EQ(
      rootGraph.GetProgramObject(),
      reinterpret_cast<uintptr_t>(&gammaTest.graph().GetRootRegion()));
  EXPECT_EQ(rootGraph.NumNodes(), 1u);       // Only the lambda node for "f"
  EXPECT_EQ(rootGraph.NumResultNodes(), 1u); // Exporting the function "f"
  auto & lambdaNode = *assertedCast<InOutNode>(&rootGraph.GetNode(0));

  // The lambda only has one output, and a single subgraph
  EXPECT_EQ(lambdaNode.GetLabel(), gammaTest.lambda->DebugString());
  EXPECT_EQ(lambdaNode.NumInputPorts(), 0u);
  EXPECT_EQ(lambdaNode.NumOutputPorts(), 1u);
  EXPECT_EQ(lambdaNode.NumSubgraphs(), 0u);

  // Check that the output of the lambda leads to a graph export
  auto & lambdaConnections = lambdaNode.GetOutputPort(0).GetConnections();
  EXPECT_EQ(lambdaConnections.size(), 1u);
  auto & graphExport = lambdaConnections.front()->GetTo().GetNode();
  EXPECT_EQ(graphExport.GetLabel(), "export[f]");
}

TEST(DotWriterTests, TestTypeGraph)
{
  using namespace jlm::llvm;
  using namespace jlm::util;
  using namespace jlm::util::graph;

  // Arrange
  GammaTest gammaTest;
  auto ptrType = PointerType::Create();
  auto bit32Type = jlm::rvsdg::BitType::Create(32);
  auto memType = MemoryStateType::Create();

  // Act
  Writer writer;
  LlvmDotWriter dotWriter;
  dotWriter.WriteGraphs(writer, gammaTest.graph().GetRootRegion(), true);

  writer.Finalize();
  writer.outputAllGraphs(std::cout, OutputFormat::Dot);

  // Assert
  auto & typeGraph = writer.GetGraph(0);
  EXPECT_EQ(typeGraph.GetProgramObject(), 0u);

  // Check that nodes exist for the given types
  [[maybe_unused]] auto & ptrNode = typeGraph.GetFromProgramObject<Node>(*ptrType);
  [[maybe_unused]] auto & bit32Node = typeGraph.GetFromProgramObject<Node>(*ptrType);
  auto & memNode = typeGraph.GetFromProgramObject<Node>(*memType);

  // Check that the rightmost argument of the function references the memNode type
  auto & fGraph = writer.GetGraph(2);
  EXPECT_EQ(writer.GetElementFromProgramObject(*gammaTest.lambda->subregion()), &fGraph);
  EXPECT_EQ(fGraph.GetArgumentNode(5).GetAttributeGraphElement("type"), &memNode);
}
