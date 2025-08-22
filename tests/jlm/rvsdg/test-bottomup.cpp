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
  jlm::rvsdg::Graph graph;
  auto vtype = jlm::tests::ValueType::Create();
  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, {});
  auto n2 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { vtype });

  jlm::rvsdg::GraphExport::Create(*n2->output(0), "dummy");

  bool n1_visited = false;
  bool n2_visited = false;
  for (const auto & node : jlm::rvsdg::BottomUpTraverser(&graph.GetRootRegion()))
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
  jlm::rvsdg::Graph graph;
  auto type = jlm::tests::ValueType::Create();
  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type, type });
  auto n2 = jlm::tests::TestOperation::create(
      &graph.GetRootRegion(),
      { n1->output(0), n1->output(1) },
      { type });

  jlm::rvsdg::GraphExport::Create(*n2->output(0), "dummy");

  {
    const jlm::rvsdg::Node * tmp = nullptr;
    jlm::rvsdg::BottomUpTraverser trav(&graph.GetRootRegion());
    tmp = trav.next();
    assert(tmp == n2);
    tmp = trav.next();
    assert(tmp == n1);
    tmp = trav.next();
    assert(tmp == 0);
  }

  assert(!graph.GetRootRegion().HasActiveTrackers());
}

static void
test_order_enforcement_traversal()
{
  jlm::rvsdg::Graph graph;
  auto type = jlm::tests::ValueType::Create();
  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type, type });
  auto n2 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1->output(0) }, { type });
  auto n3 = jlm::tests::TestOperation::create(
      &graph.GetRootRegion(),
      { n2->output(0), n1->output(1) },
      { type });

  {
    const jlm::rvsdg::Node * tmp = nullptr;
    jlm::rvsdg::BottomUpTraverser trav(&graph.GetRootRegion());

    tmp = trav.next();
    assert(tmp == n3);
    tmp = trav.next();
    assert(tmp == n2);
    tmp = trav.next();
    assert(tmp == n1);
    tmp = trav.next();
    assert(tmp == nullptr);
  }

  assert(!graph.GetRootRegion().HasActiveTrackers());
}

static void
test_main()
{
  test_initialization();
  test_basic_traversal();
  test_order_enforcement_traversal();
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-bottomup", test_main)
