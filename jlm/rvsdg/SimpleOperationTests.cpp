/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(SimpleOperationTests, NormalizeSimpleOperationCne_NodesWithoutOperands)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = TestType::createValueType();
  const auto stateType = TestType::createStateType();

  auto & nullaryValueNode1 = CreateOpNode<TestNullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryValueNode2 = CreateOpNode<TestNullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryStateNode1 = CreateOpNode<TestNullaryOperation>(graph.GetRootRegion(), stateType);
  auto & nullaryStateNode2 = CreateOpNode<TestNullaryOperation>(graph.GetRootRegion(), stateType);

  auto & exNullaryValueNode1 = GraphExport::Create(*nullaryValueNode1.output(0), "nvn1");
  auto & exNullaryValueNode2 = GraphExport::Create(*nullaryValueNode2.output(0), "nvn2");
  auto & exNullaryStateNode1 = GraphExport::Create(*nullaryStateNode1.output(0), "nsn1");
  auto & exNullaryStateNode2 = GraphExport::Create(*nullaryStateNode2.output(0), "nsn2");

  view(graph, stdout);

  // Act
  auto NormalizeCne = [&](const SimpleOperation & operation, const std::vector<Output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  // Act
  ReduceNode<SimpleOperation>(
      NormalizeCne,
      *TryGetOwnerNode<SimpleNode>(*exNullaryValueNode1.origin()));
  ReduceNode<SimpleOperation>(
      NormalizeCne,
      *TryGetOwnerNode<SimpleNode>(*exNullaryValueNode2.origin()));
  ReduceNode<SimpleOperation>(
      NormalizeCne,
      *TryGetOwnerNode<SimpleNode>(*exNullaryStateNode1.origin()));
  ReduceNode<SimpleOperation>(
      NormalizeCne,
      *TryGetOwnerNode<SimpleNode>(*exNullaryStateNode2.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  EXPECT_EQ(exNullaryValueNode1.origin(), exNullaryValueNode2.origin());
  EXPECT_EQ(exNullaryStateNode1.origin(), exNullaryStateNode2.origin());
  EXPECT_NE(exNullaryValueNode1.origin(), exNullaryStateNode2.origin());
}

TEST(SimpleOperationTests, NormalizeSimpleOperationCne_NodesWithOperands)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = TestType::createValueType();
  const auto stateType = TestType::createStateType();

  auto v1 = &GraphImport::Create(graph, valueType, "v1");
  auto s1 = &GraphImport::Create(graph, stateType, "s1");

  auto & valueNode1 = CreateOpNode<TestUnaryOperation>({ v1 }, valueType, valueType);
  auto & valueNode2 = CreateOpNode<TestUnaryOperation>({ v1 }, valueType, valueType);
  auto & stateNode1 = CreateOpNode<TestUnaryOperation>({ s1 }, stateType, stateType);
  auto & stateNode2 = CreateOpNode<TestUnaryOperation>({ s1 }, stateType, stateType);

  auto & exValueNode1 = GraphExport::Create(*valueNode1.output(0), "nvn1");
  auto & exValueNode2 = GraphExport::Create(*valueNode2.output(0), "nvn2");
  auto & exStateNode1 = GraphExport::Create(*stateNode1.output(0), "nsn1");
  auto & exStateNode2 = GraphExport::Create(*stateNode2.output(0), "nsn2");

  view(graph, stdout);

  // Act
  auto NormalizeCne = [&](const SimpleOperation & operation, const std::vector<Output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  // Act
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*exValueNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*exValueNode2.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*exStateNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*exStateNode2.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  EXPECT_EQ(exValueNode1.origin(), exValueNode2.origin());
  EXPECT_EQ(exStateNode1.origin(), exStateNode2.origin());
  EXPECT_NE(exValueNode1.origin(), exStateNode2.origin());
}

TEST(SimpleOperationTests, NormalizeSimpleOperationCne_Mixed)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  auto NormalizeCne =
      [&](const SimpleOperation & operation, const std::vector<jlm::rvsdg::Output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  auto valueType = TestType::createValueType();

  auto i = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i");

  auto o1 = TestOperation::createNode(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto o2 = TestOperation::createNode(&graph.GetRootRegion(), { i }, { valueType })->output(0);
  auto o3 = TestOperation::createNode(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto o4 = TestOperation::createNode(&graph.GetRootRegion(), { i }, { valueType })->output(0);

  auto & e1 = GraphExport::Create(*o1, "o1");
  auto & e2 = GraphExport::Create(*o2, "o2");
  auto & e3 = GraphExport::Create(*o3, "o3");
  auto & e4 = GraphExport::Create(*o4, "o4");

  // Act & Assert
  ReduceNode<TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e1.origin()));
  ReduceNode<TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e2.origin()));
  ReduceNode<TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e3.origin()));
  ReduceNode<TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e4.origin()));

  EXPECT_EQ(e1.origin(), e3.origin());
  EXPECT_EQ(e2.origin(), e4.origin());

  auto o5 = TestOperation::createNode(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto & e5 = GraphExport::Create(*o5, "o5");
  ReduceNode<TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e5.origin()));
  EXPECT_EQ(e5.origin(), e1.origin());

  auto o6 = TestOperation::createNode(&graph.GetRootRegion(), { i }, { valueType })->output(0);
  auto & e6 = GraphExport::Create(*o6, "o6");
  ReduceNode<TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e6.origin()));
  EXPECT_EQ(e6.origin(), e2.origin());

  auto o7 = TestOperation::createNode(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto & e7 = GraphExport::Create(*o7, "o7");
  EXPECT_NE(e7.origin(), e1.origin());

  ReduceNode<TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e7.origin()));
  EXPECT_EQ(e7.origin(), e1.origin());
}

TEST(SimpleOperationTests, NormalizeSimpleOperationCne_Failure)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = TestType::createValueType();
  const auto stateType = TestType::createStateType();

  auto v1 = &GraphImport::Create(graph, valueType, "v1");
  auto s1 = &GraphImport::Create(graph, stateType, "s1");

  auto & nullaryValueNode = CreateOpNode<TestNullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryStateNode = CreateOpNode<TestNullaryOperation>(graph.GetRootRegion(), stateType);
  auto & unaryValueNode = CreateOpNode<TestUnaryOperation>({ v1 }, valueType, valueType);
  auto & unaryStateNode = CreateOpNode<TestUnaryOperation>({ s1 }, stateType, stateType);

  auto & exNullaryValueNode = GraphExport::Create(*nullaryValueNode.output(0), "nvn1");
  auto & exNullaryStateNode = GraphExport::Create(*nullaryStateNode.output(0), "nvn2");
  auto & exUnaryValueNode = GraphExport::Create(*unaryValueNode.output(0), "nsn1");
  auto & exUnaryStateNode = GraphExport::Create(*unaryStateNode.output(0), "nsn2");

  view(graph, stdout);

  // Act
  auto NormalizeCne = [&](const SimpleOperation & operation, const std::vector<Output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  // Act
  ReduceNode<SimpleOperation>(
      NormalizeCne,
      *TryGetOwnerNode<SimpleNode>(*exNullaryValueNode.origin()));
  ReduceNode<SimpleOperation>(
      NormalizeCne,
      *TryGetOwnerNode<SimpleNode>(*exNullaryStateNode.origin()));
  ReduceNode<SimpleOperation>(
      NormalizeCne,
      *TryGetOwnerNode<SimpleNode>(*exUnaryValueNode.origin()));
  ReduceNode<SimpleOperation>(
      NormalizeCne,
      *TryGetOwnerNode<SimpleNode>(*exUnaryStateNode.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  EXPECT_EQ(TryGetOwnerNode<Node>(*exNullaryValueNode.origin()), &nullaryValueNode);
  EXPECT_EQ(TryGetOwnerNode<Node>(*exNullaryStateNode.origin()), &nullaryStateNode);
  EXPECT_EQ(TryGetOwnerNode<Node>(*exUnaryValueNode.origin()), &unaryValueNode);
  EXPECT_EQ(TryGetOwnerNode<Node>(*exUnaryStateNode.origin()), &unaryStateNode);
}
