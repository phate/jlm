/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

static void
test_main()
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

  auto valueType = jlm::tests::ValueType::Create();

  auto i = &jlm::tests::GraphImport::Create(graph, valueType, "i");

  auto o1 = jlm::tests::test_op::create(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto o2 = jlm::tests::test_op::create(&graph.GetRootRegion(), { i }, { valueType })->output(0);
  auto o3 = jlm::tests::create_testop(&graph.GetRootRegion(), {}, { valueType })[0];
  auto o4 = jlm::tests::create_testop(&graph.GetRootRegion(), { i }, { valueType })[0];

  auto & e1 = jlm::tests::GraphExport::Create(*o1, "o1");
  auto & e2 = jlm::tests::GraphExport::Create(*o2, "o2");
  auto & e3 = jlm::tests::GraphExport::Create(*o3, "o3");
  auto & e4 = jlm::tests::GraphExport::Create(*o4, "o4");

  // Act & Assert
  ReduceNode<jlm::tests::test_op>(NormalizeCne, *TryGetOwnerNode<Node>(*e1.origin()));
  ReduceNode<jlm::tests::test_op>(NormalizeCne, *TryGetOwnerNode<Node>(*e2.origin()));
  ReduceNode<jlm::tests::test_op>(NormalizeCne, *TryGetOwnerNode<Node>(*e3.origin()));
  ReduceNode<jlm::tests::test_op>(NormalizeCne, *TryGetOwnerNode<Node>(*e4.origin()));

  assert(e1.origin() == e3.origin());
  assert(e2.origin() == e4.origin());

  auto o5 = jlm::tests::create_testop(&graph.GetRootRegion(), {}, { valueType })[0];
  auto & e5 = jlm::tests::GraphExport::Create(*o5, "o5");
  ReduceNode<jlm::tests::test_op>(NormalizeCne, *TryGetOwnerNode<Node>(*e5.origin()));
  assert(e5.origin() == e1.origin());

  auto o6 = jlm::tests::create_testop(&graph.GetRootRegion(), { i }, { valueType })[0];
  auto & e6 = jlm::tests::GraphExport::Create(*o6, "o6");
  ReduceNode<jlm::tests::test_op>(NormalizeCne, *TryGetOwnerNode<Node>(*e6.origin()));
  assert(e6.origin() == e2.origin());

  auto o7 = jlm::tests::create_testop(&graph.GetRootRegion(), {}, { valueType })[0];
  auto & e7 = jlm::tests::GraphExport::Create(*o7, "o7");
  assert(e7.origin() != e1.origin());

  ReduceNode<jlm::tests::test_op>(NormalizeCne, *TryGetOwnerNode<Node>(*e7.origin()));
  assert(e7.origin() == e1.origin());
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-cse", test_main)
