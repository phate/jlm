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

  auto i = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i");

  auto o1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto o2 =
      jlm::tests::TestOperation::create(&graph.GetRootRegion(), { i }, { valueType })->output(0);
  auto o3 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto o4 =
      jlm::tests::TestOperation::create(&graph.GetRootRegion(), { i }, { valueType })->output(0);

  auto & e1 = GraphExport::Create(*o1, "o1");
  auto & e2 = GraphExport::Create(*o2, "o2");
  auto & e3 = GraphExport::Create(*o3, "o3");
  auto & e4 = GraphExport::Create(*o4, "o4");

  // Act & Assert
  ReduceNode<jlm::tests::TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e1.origin()));
  ReduceNode<jlm::tests::TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e2.origin()));
  ReduceNode<jlm::tests::TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e3.origin()));
  ReduceNode<jlm::tests::TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e4.origin()));

  assert(e1.origin() == e3.origin());
  assert(e2.origin() == e4.origin());

  auto o5 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto & e5 = GraphExport::Create(*o5, "o5");
  ReduceNode<jlm::tests::TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e5.origin()));
  assert(e5.origin() == e1.origin());

  auto o6 =
      jlm::tests::TestOperation::create(&graph.GetRootRegion(), { i }, { valueType })->output(0);
  auto & e6 = GraphExport::Create(*o6, "o6");
  ReduceNode<jlm::tests::TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e6.origin()));
  assert(e6.origin() == e2.origin());

  auto o7 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { valueType })->output(0);
  auto & e7 = GraphExport::Create(*o7, "o7");
  assert(e7.origin() != e1.origin());

  ReduceNode<jlm::tests::TestOperation>(NormalizeCne, *TryGetOwnerNode<SimpleNode>(*e7.origin()));
  assert(e7.origin() == e1.origin());
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-cse", test_main)
