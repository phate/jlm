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
  ReduceNode<SimpleOperation>(NormalizeCne, *output::GetNode(*exNullaryValueNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *output::GetNode(*exNullaryValueNode2.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *output::GetNode(*exNullaryStateNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *output::GetNode(*exNullaryStateNode2.origin()));
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
  ReduceNode<SimpleOperation>(NormalizeCne, *output::GetNode(*exValueNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *output::GetNode(*exValueNode2.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *output::GetNode(*exStateNode1.origin()));
  ReduceNode<SimpleOperation>(NormalizeCne, *output::GetNode(*exStateNode2.origin()));
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
