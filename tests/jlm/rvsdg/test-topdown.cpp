/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/traverser.hpp>

static void
test_initialization()
{
  auto vtype = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto i = &jlm::tests::GraphImport::Create(graph, vtype, "i");

  auto constant = jlm::tests::test_op::create(graph.root(), {}, { vtype });
  auto unary = jlm::tests::test_op::create(graph.root(), { i }, { vtype });
  auto binary = jlm::tests::test_op::create(graph.root(), { i, unary->output(0) }, { vtype });

  jlm::tests::GraphExport::Create(*constant->output(0), "c");
  jlm::tests::GraphExport::Create(*unary->output(0), "u");
  jlm::tests::GraphExport::Create(*binary->output(0), "b");

  bool unary_visited = false;
  bool binary_visited = false;
  bool constant_visited = false;
  for (const auto & node : jlm::rvsdg::topdown_traverser(graph.root()))
  {
    if (node == unary)
      unary_visited = true;
    if (node == constant)
      constant_visited = true;
    if (node == binary && unary_visited)
      binary_visited = true;
  }

  assert(unary_visited);
  assert(binary_visited);
  assert(constant_visited);
}

static void
test_basic_traversal()
{
  jlm::rvsdg::graph graph;
  auto type = jlm::tests::valuetype::Create();

  auto n1 = jlm::tests::test_op::create(graph.root(), {}, { type, type });
  auto n2 = jlm::tests::test_op::create(graph.root(), { n1->output(0), n1->output(1) }, { type });

  jlm::tests::GraphExport::Create(*n2->output(0), "dummy");

  {
    jlm::rvsdg::node * tmp;
    jlm::rvsdg::topdown_traverser trav(graph.root());

    tmp = trav.next();
    assert(tmp == n1);
    tmp = trav.next();
    assert(tmp == n2);
    tmp = trav.next();
    assert(tmp == nullptr);
  }

  assert(!has_active_trackers(&graph));
}

static void
test_order_enforcement_traversal()
{
  jlm::rvsdg::graph graph;
  auto type = jlm::tests::valuetype::Create();

  auto n1 = jlm::tests::test_op::create(graph.root(), {}, { type, type });
  auto n2 = jlm::tests::test_op::create(graph.root(), { n1->output(0) }, { type });
  auto n3 = jlm::tests::test_op::create(graph.root(), { n2->output(0), n1->output(1) }, { type });

  {
    jlm::rvsdg::node * tmp;
    jlm::rvsdg::topdown_traverser trav(graph.root());

    tmp = trav.next();
    assert(tmp == n1);
    tmp = trav.next();
    assert(tmp == n2);
    tmp = trav.next();
    assert(tmp == n3);
    tmp = trav.next();
    assert(tmp == nullptr);
  }

  assert(!has_active_trackers(&graph));
}

static void
test_traversal_insertion()
{
  jlm::rvsdg::graph graph;
  auto type = jlm::tests::valuetype::Create();

  auto n1 = jlm::tests::test_op::create(graph.root(), {}, { type, type });
  auto n2 = jlm::tests::test_op::create(graph.root(), { n1->output(0), n1->output(1) }, { type });

  jlm::tests::GraphExport::Create(*n2->output(0), "dummy");

  {
    jlm::rvsdg::node * node;
    jlm::rvsdg::topdown_traverser trav(graph.root());

    node = trav.next();
    assert(node == n1);

    /* At this point, n1 has been visited, now create some nodes */

    auto n3 = jlm::tests::test_op::create(graph.root(), {}, { type });
    auto n4 = jlm::tests::test_op::create(graph.root(), { n3->output(0) }, {});
    auto n5 = jlm::tests::test_op::create(graph.root(), { n2->output(0) }, {});

    /*
      The newly created nodes n3 and n4 will not be visited,
      as they were not created as descendants of unvisited
      nodes. n5 must be visited, as n2 has not been visited yet.
    */

    bool visited_n2 = false, visited_n3 = false, visited_n4 = false, visited_n5 = false;
    for (;;)
    {
      node = trav.next();
      if (!node)
        break;
      if (node == n2)
        visited_n2 = true;
      if (node == n3)
        visited_n3 = true;
      if (node == n4)
        visited_n4 = true;
      if (node == n5)
        visited_n5 = true;
    }

    assert(visited_n2);
    assert(!visited_n3);
    assert(!visited_n4);
    assert(visited_n5);
  }

  assert(!has_active_trackers(&graph));
}

static void
test_mutable_traverse()
{
  auto test = [](jlm::rvsdg::graph * graph,
                 jlm::rvsdg::node * n1,
                 jlm::rvsdg::node * n2,
                 jlm::rvsdg::node * n3)
  {
    bool seen_n1 = false;
    bool seen_n2 = false;
    bool seen_n3 = false;

    for (const auto & tmp : jlm::rvsdg::topdown_traverser(graph->root()))
    {
      seen_n1 = seen_n1 || (tmp == n1);
      seen_n2 = seen_n2 || (tmp == n2);
      seen_n3 = seen_n3 || (tmp == n3);
      if (n3->input(0)->origin() == n1->output(0))
        n3->input(0)->divert_to(n2->output(0));
      else
        n3->input(0)->divert_to(n1->output(0));
    }

    assert(seen_n1);
    assert(seen_n2);
    assert(seen_n3);
  };

  jlm::rvsdg::graph graph;
  auto type = jlm::tests::valuetype::Create();
  auto n1 = jlm::tests::test_op::create(graph.root(), {}, { type });
  auto n2 = jlm::tests::test_op::create(graph.root(), {}, { type });
  auto n3 = jlm::tests::test_op::create(graph.root(), { n1->output(0) }, {});

  test(&graph, n1, n2, n3);
  test(&graph, n1, n2, n3);
}

static int
test_main(void)
{
  test_initialization();
  test_basic_traversal();
  test_order_enforcement_traversal();
  test_traversal_insertion();
  test_mutable_traverse();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-topdown", test_main)
