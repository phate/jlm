/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <assert.h>
#include <stdio.h>

#include <jlm/rvsdg/view.hpp>

static bool
region_contains_node(const jlm::rvsdg::region * region, const jlm::rvsdg::node * n)
{
  for (const auto & node : region->nodes)
  {
    if (&node == n)
      return true;
  }

  return false;
}

static int
test_recursive_prune()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto t = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto imp = &jlm::tests::GraphImport::Create(graph, t, "i");

  auto n1 = jlm::tests::test_op::create(graph.root(), { imp }, { t });
  auto n2 = jlm::tests::test_op::create(graph.root(), { imp }, { t });

  auto n3 = jlm::tests::structural_node::create(graph.root(), 1);
  structural_input::create(n3, imp, t);
  auto & a1 = TestGraphArgument::Create(*n3->subregion(0), nullptr, t);
  auto n4 = jlm::tests::test_op::create(n3->subregion(0), { &a1 }, { t });
  auto n5 = jlm::tests::test_op::create(n3->subregion(0), { &a1 }, { t });
  TestGraphResult::Create(*n4->output(0), nullptr);
  auto o1 = structural_output::create(n3, t);

  auto n6 = jlm::tests::structural_node::create(n3->subregion(0), 1);

  jlm::tests::GraphExport::Create(*n2->output(0), "n2");
  jlm::tests::GraphExport::Create(*o1, "n3");

  jlm::rvsdg::view(graph.root(), stdout);
  graph.prune();
  jlm::rvsdg::view(graph.root(), stdout);

  assert(!region_contains_node(graph.root(), n1));
  assert(region_contains_node(graph.root(), n2));
  assert(region_contains_node(graph.root(), n3));
  assert(region_contains_node(n3->subregion(0), n4));
  assert(!region_contains_node(n3->subregion(0), n5));
  assert(!region_contains_node(n3->subregion(0), n6));

  return 0;
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-graph_prune", test_recursive_prune)

static int
test_empty_graph_pruning(void)
{
  jlm::rvsdg::graph graph;

  jlm::rvsdg::view(graph.root(), stdout);

  graph.prune();

  assert(graph.root()->nnodes() == 0);

  jlm::rvsdg::view(graph.root(), stdout);

  return 0;
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-empty_graph_pruning", test_empty_graph_pruning)

static int
test_prune_replace(void)
{
  using namespace jlm::rvsdg;

  auto type = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto n1 = jlm::tests::test_op::create(graph.root(), {}, { type });
  auto n2 = jlm::tests::test_op::create(graph.root(), { n1->output(0) }, { type });
  auto n3 = jlm::tests::test_op::create(graph.root(), { n2->output(0) }, { type });

  jlm::tests::GraphExport::Create(*n2->output(0), "n2");
  jlm::tests::GraphExport::Create(*n3->output(0), "n3");

  auto n4 = jlm::tests::test_op::create(graph.root(), { n1->output(0) }, { type });

  n2->output(0)->divert_users(n4->output(0));
  assert(n2->output(0)->nusers() == 0);

  graph.prune();

  assert(!region_contains_node(graph.root(), n2));

  return 0;
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-prune-replace", test_prune_replace)

static int
test_graph(void)
{
  using namespace jlm::rvsdg;

  auto type = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;

  auto n1 = jlm::tests::test_op::create(graph.root(), {}, { type });
  assert(n1);
  assert(n1->depth() == 0);

  auto n2 = jlm::tests::test_op::create(graph.root(), { n1->output(0) }, {});
  assert(n2);
  assert(n2->depth() == 1);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-graph", test_graph)

static int
Copy()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto & argument = TestGraphArgument::Create(*graph.root(), nullptr, valueType);
  auto node = test_op::create(graph.root(), { &argument }, { valueType });
  TestGraphResult::Create(*node->output(0), nullptr);

  // Act
  auto newGraph = graph.copy();

  // Assert
  assert(newGraph->root()->narguments() == 1);
  auto copiedArgument = newGraph->root()->argument(0);
  assert(is<TestGraphArgument>(copiedArgument));

  assert(newGraph->root()->nnodes() == 1);
  auto copiedNode = newGraph->root()->nodes.first();
  assert(copiedNode->ninputs() == 1 && copiedNode->noutputs() == 1);
  assert(copiedNode->input(0)->origin() == copiedArgument);

  assert(newGraph->root()->nresults() == 1);
  auto copiedResult = newGraph->root()->result(0);
  assert(is<TestGraphResult>(*copiedResult));
  assert(copiedResult->origin() == copiedNode->output(0));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-graph-Copy", Copy)
