/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

static bool
region_contains_node(const jlm::rvsdg::Region * region, const jlm::rvsdg::Node * n)
{
  for (const auto & node : region->Nodes())
  {
    if (&node == n)
      return true;
  }

  return false;
}

static void
test_recursive_prune()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto t = jlm::tests::ValueType::Create();

  Graph graph;
  auto imp = &jlm::tests::GraphImport::Create(graph, t, "i");

  auto n1 = TestOperation::create(&graph.GetRootRegion(), { imp }, { t });
  auto n2 = TestOperation::create(&graph.GetRootRegion(), { imp }, { t });

  auto n3 = jlm::tests::structural_node::create(&graph.GetRootRegion(), 1);
  StructuralInput::create(n3, imp, t);
  auto & a1 = TestGraphArgument::Create(*n3->subregion(0), nullptr, t);
  auto n4 = TestOperation::create(n3->subregion(0), { &a1 }, { t });
  auto n5 = TestOperation::create(n3->subregion(0), { &a1 }, { t });
  TestGraphResult::Create(*n4->output(0), nullptr);
  auto o1 = StructuralOutput::create(n3, t);

  auto n6 = jlm::tests::structural_node::create(n3->subregion(0), 1);

  jlm::tests::GraphExport::Create(*n2->output(0), "n2");
  jlm::tests::GraphExport::Create(*o1, "n3");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);
  graph.PruneNodes();
  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  assert(!region_contains_node(&graph.GetRootRegion(), n1));
  assert(region_contains_node(&graph.GetRootRegion(), n2));
  assert(region_contains_node(&graph.GetRootRegion(), n3));
  assert(region_contains_node(n3->subregion(0), n4));
  assert(!region_contains_node(n3->subregion(0), n5));
  assert(!region_contains_node(n3->subregion(0), n6));
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-graph_prune", test_recursive_prune)

static void
test_empty_graph_pruning()
{
  jlm::rvsdg::Graph graph;

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  graph.PruneNodes();

  assert(graph.GetRootRegion().nnodes() == 0);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-empty_graph_pruning", test_empty_graph_pruning)

static void
test_prune_replace()
{
  using namespace jlm::rvsdg;

  auto type = jlm::tests::ValueType::Create();

  Graph graph;
  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type });
  auto n2 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1->output(0) }, { type });
  auto n3 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n2->output(0) }, { type });

  jlm::tests::GraphExport::Create(*n2->output(0), "n2");
  jlm::tests::GraphExport::Create(*n3->output(0), "n3");

  auto n4 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1->output(0) }, { type });

  n2->output(0)->divert_users(n4->output(0));
  assert(n2->output(0)->nusers() == 0);

  graph.PruneNodes();

  assert(!region_contains_node(&graph.GetRootRegion(), n2));
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-prune-replace", test_prune_replace)

static void
test_graph()
{
  using namespace jlm::rvsdg;

  auto type = jlm::tests::ValueType::Create();

  Graph graph;

  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type });
  assert(n1);
  assert(n1->depth() == 0);

  auto n2 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1->output(0) }, {});
  assert(n2);
  assert(n2->depth() == 1);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-graph", test_graph)

static void
Copy()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph graph;
  auto & argument = TestGraphArgument::Create(graph.GetRootRegion(), nullptr, valueType);
  auto node = TestOperation::create(&graph.GetRootRegion(), { &argument }, { valueType });
  TestGraphResult::Create(*node->output(0), nullptr);

  // Act
  auto newGraph = graph.Copy();

  // Assert
  assert(newGraph->GetRootRegion().narguments() == 1);
  auto copiedArgument = newGraph->GetRootRegion().argument(0);
  assert(is<TestGraphArgument>(copiedArgument));

  assert(newGraph->GetRootRegion().nnodes() == 1);
  auto copiedNode = newGraph->GetRootRegion().Nodes().begin().ptr();
  assert(copiedNode->ninputs() == 1 && copiedNode->noutputs() == 1);
  assert(copiedNode->input(0)->origin() == copiedArgument);

  assert(newGraph->GetRootRegion().nresults() == 1);
  auto copiedResult = newGraph->GetRootRegion().result(0);
  assert(is<TestGraphResult>(*copiedResult));
  assert(copiedResult->origin() == copiedNode->output(0));
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-graph-Copy", Copy)
