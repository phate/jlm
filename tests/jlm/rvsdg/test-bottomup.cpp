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
  jlm::rvsdg::graph graph;
  auto vtype = jlm::tests::valuetype::Create();
  auto n1 = jlm::tests::test_op::create(graph.root(), {}, {});
  auto n2 = jlm::tests::test_op::create(graph.root(), {}, { vtype });

  jlm::tests::GraphExport::Create(*n2->output(0), "dummy");

  bool n1_visited = false;
  bool n2_visited = false;
  for (const auto & node : jlm::rvsdg::bottomup_traverser(graph.root()))
  {
    if (node == n1)
      n1_visited = true;
    if (node == n2)
      n2_visited = true;
  }

  assert(n1_visited);
  assert(n2_visited);
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
    jlm::rvsdg::bottomup_traverser trav(graph.root());
    tmp = trav.next();
    assert(tmp == n2);
    tmp = trav.next();
    assert(tmp == n1);
    tmp = trav.next();
    assert(tmp == 0);
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

  jlm::rvsdg::node * tmp;
  {
    jlm::rvsdg::bottomup_traverser trav(graph.root());

    tmp = trav.next();
    assert(tmp == n3);
    tmp = trav.next();
    assert(tmp == n2);
    tmp = trav.next();
    assert(tmp == n1);
    tmp = trav.next();
    assert(tmp == nullptr);
  }

  assert(!has_active_trackers(&graph));
}

static int
test_main()
{
  test_initialization();
  test_basic_traversal();
  test_order_enforcement_traversal();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-bottomup", test_main)
