/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

static void
NormalizeSimpleOperationCne_NodesWithoutOperands()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = TestType::createValueType();
  const auto stateType = TestType::createStateType();

  auto & nullaryValueNode1 =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryValueNode2 =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryStateNode1 =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), stateType);
  auto & nullaryStateNode2 =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), stateType);

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
  assert(exNullaryValueNode1.origin() == exNullaryValueNode2.origin());
  assert(exNullaryStateNode1.origin() == exNullaryStateNode2.origin());
  assert(exNullaryValueNode1.origin() != exNullaryStateNode2.origin());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/NormalizeSimpleOperationCne_NodesWithoutOperands",
    NormalizeSimpleOperationCne_NodesWithoutOperands)

static void
NormalizeSimpleOperationCne_NodesWithOperands()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = TestType::createValueType();
  const auto stateType = TestType::createStateType();

  auto v1 = &GraphImport::Create(graph, valueType, "v1");
  auto s1 = &GraphImport::Create(graph, stateType, "s1");

  auto & valueNode1 = CreateOpNode<jlm::tests::TestUnaryOperation>({ v1 }, valueType, valueType);
  auto & valueNode2 = CreateOpNode<jlm::tests::TestUnaryOperation>({ v1 }, valueType, valueType);
  auto & stateNode1 = CreateOpNode<jlm::tests::TestUnaryOperation>({ s1 }, stateType, stateType);
  auto & stateNode2 = CreateOpNode<jlm::tests::TestUnaryOperation>({ s1 }, stateType, stateType);

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
  assert(exValueNode1.origin() == exValueNode2.origin());
  assert(exStateNode1.origin() == exStateNode2.origin());
  assert(exValueNode1.origin() != exStateNode2.origin());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/NormalizeSimpleOperationCne_NodesWithOperands",
    NormalizeSimpleOperationCne_NodesWithOperands)

static void
NormalizeSimpleOperationCne_Failure()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = TestType::createValueType();
  const auto stateType = TestType::createStateType();

  auto v1 = &GraphImport::Create(graph, valueType, "v1");
  auto s1 = &GraphImport::Create(graph, stateType, "s1");

  auto & nullaryValueNode =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryStateNode =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), stateType);
  auto & unaryValueNode =
      CreateOpNode<jlm::tests::TestUnaryOperation>({ v1 }, valueType, valueType);
  auto & unaryStateNode =
      CreateOpNode<jlm::tests::TestUnaryOperation>({ s1 }, stateType, stateType);

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
  assert(TryGetOwnerNode<Node>(*exNullaryValueNode.origin()) == &nullaryValueNode);
  assert(TryGetOwnerNode<Node>(*exNullaryStateNode.origin()) == &nullaryStateNode);
  assert(TryGetOwnerNode<Node>(*exUnaryValueNode.origin()) == &unaryValueNode);
  assert(TryGetOwnerNode<Node>(*exUnaryStateNode.origin()) == &unaryStateNode);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/NormalizeSimpleOperationCne_Failure",
    NormalizeSimpleOperationCne_Failure)
