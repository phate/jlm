/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/view.hpp>

static int
NormalizeSimpleOperationCne_NodesWithoutOperands()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = jlm::tests::valuetype::Create();
  const auto stateType = jlm::tests::statetype::Create();

  auto & nullaryValueNode1 =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryValueNode2 =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryStateNode1 =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), stateType);
  auto & nullaryStateNode2 =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), stateType);

  auto & exNullaryValueNode1 =
      jlm::tests::GraphExport::Create(*nullaryValueNode1.output(0), "nvn1");
  auto & exNullaryValueNode2 =
      jlm::tests::GraphExport::Create(*nullaryValueNode2.output(0), "nvn2");
  auto & exNullaryStateNode1 =
      jlm::tests::GraphExport::Create(*nullaryStateNode1.output(0), "nsn1");
  auto & exNullaryStateNode2 =
      jlm::tests::GraphExport::Create(*nullaryStateNode2.output(0), "nsn2");

  view(graph, stdout);

  // Act
  auto NormalizeCne = [&](const SimpleOperation & operation, const std::vector<output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  // Act
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exNullaryValueNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exNullaryValueNode2.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exNullaryStateNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exNullaryStateNode2.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(exNullaryValueNode1.origin() == exNullaryValueNode2.origin());
  assert(exNullaryStateNode1.origin() == exNullaryStateNode2.origin());
  assert(exNullaryValueNode1.origin() != exNullaryStateNode2.origin());

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/NormalizeSimpleOperationCne_NodesWithoutOperands",
    NormalizeSimpleOperationCne_NodesWithoutOperands)

static int
NormalizeSimpleOperationCne_NodesWithOperands()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = jlm::tests::valuetype::Create();
  const auto stateType = jlm::tests::statetype::Create();

  auto v1 = &jlm::tests::GraphImport::Create(graph, valueType, "v1");
  auto s1 = &jlm::tests::GraphImport::Create(graph, stateType, "s1");

  auto & valueNode1 = CreateOpNode<jlm::tests::unary_op>({ v1 }, valueType, valueType);
  auto & valueNode2 = CreateOpNode<jlm::tests::unary_op>({ v1 }, valueType, valueType);
  auto & stateNode1 = CreateOpNode<jlm::tests::unary_op>({ s1 }, stateType, stateType);
  auto & stateNode2 = CreateOpNode<jlm::tests::unary_op>({ s1 }, stateType, stateType);

  auto & exValueNode1 = jlm::tests::GraphExport::Create(*valueNode1.output(0), "nvn1");
  auto & exValueNode2 = jlm::tests::GraphExport::Create(*valueNode2.output(0), "nvn2");
  auto & exStateNode1 = jlm::tests::GraphExport::Create(*stateNode1.output(0), "nsn1");
  auto & exStateNode2 = jlm::tests::GraphExport::Create(*stateNode2.output(0), "nsn2");

  view(graph, stdout);

  // Act
  auto NormalizeCne = [&](const SimpleOperation & operation, const std::vector<output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  // Act
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exValueNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exValueNode2.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exStateNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exStateNode2.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(exValueNode1.origin() == exValueNode2.origin());
  assert(exStateNode1.origin() == exStateNode2.origin());
  assert(exValueNode1.origin() != exStateNode2.origin());

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/NormalizeSimpleOperationCne_NodesWithOperands",
    NormalizeSimpleOperationCne_NodesWithOperands)

static int
NormalizeSimpleOperationCne_Failure()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = jlm::tests::valuetype::Create();
  const auto stateType = jlm::tests::statetype::Create();

  auto v1 = &jlm::tests::GraphImport::Create(graph, valueType, "v1");
  auto s1 = &jlm::tests::GraphImport::Create(graph, stateType, "s1");

  auto & nullaryValueNode =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), valueType);
  auto & nullaryStateNode =
      CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), stateType);
  auto & unaryValueNode = CreateOpNode<jlm::tests::unary_op>({ v1 }, valueType, valueType);
  auto & unaryStateNode = CreateOpNode<jlm::tests::unary_op>({ s1 }, stateType, stateType);

  auto & exNullaryValueNode = jlm::tests::GraphExport::Create(*nullaryValueNode.output(0), "nvn1");
  auto & exNullaryStateNode = jlm::tests::GraphExport::Create(*nullaryStateNode.output(0), "nvn2");
  auto & exUnaryValueNode = jlm::tests::GraphExport::Create(*unaryValueNode.output(0), "nsn1");
  auto & exUnaryStateNode = jlm::tests::GraphExport::Create(*unaryStateNode.output(0), "nsn2");

  view(graph, stdout);

  // Act
  auto NormalizeCne = [&](const SimpleOperation & operation, const std::vector<output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  // Act
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exNullaryValueNode.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exNullaryStateNode.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exUnaryValueNode.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*exUnaryStateNode.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*exNullaryValueNode.origin()) == &nullaryValueNode);
  assert(TryGetOwnerNode<Node>(*exNullaryStateNode.origin()) == &nullaryStateNode);
  assert(TryGetOwnerNode<Node>(*exUnaryValueNode.origin()) == &unaryValueNode);
  assert(TryGetOwnerNode<Node>(*exUnaryStateNode.origin()) == &unaryStateNode);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/NormalizeSimpleOperationCne_Failure",
    NormalizeSimpleOperationCne_Failure)
