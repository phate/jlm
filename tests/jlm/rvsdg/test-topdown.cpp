/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/traverser.hpp>

static void
testInitialization()
{
  auto vtype = jlm::tests::ValueType::Create();

  jlm::rvsdg::Graph graph;
  auto i = &jlm::rvsdg::GraphImport::Create(graph, vtype, "i");

  auto constant = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { vtype });
  auto unary = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { i }, { vtype });
  auto binary =
      jlm::tests::TestOperation::create(&graph.GetRootRegion(), { i, unary->output(0) }, { vtype });

  jlm::rvsdg::GraphExport::Create(*constant->output(0), "c");
  jlm::rvsdg::GraphExport::Create(*unary->output(0), "u");
  jlm::rvsdg::GraphExport::Create(*binary->output(0), "b");

  bool unary_visited = false;
  bool binary_visited = false;
  bool constant_visited = false;
  for (const auto & node : jlm::rvsdg::TopDownTraverser(&graph.GetRootRegion()))
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

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TopdownTraverserTest-testInitialization", testInitialization);

static void
testBasicTraversal()
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
    jlm::rvsdg::TopDownTraverser trav(&graph.GetRootRegion());

    tmp = trav.next();
    assert(tmp == n1);
    tmp = trav.next();
    assert(tmp == n2);
    tmp = trav.next();
    assert(tmp == nullptr);
  }
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TopdownTraverserTest-testBasicTraversal", testBasicTraversal);

static void
testOrderEnforcement()
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
    jlm::rvsdg::TopDownTraverser trav(&graph.GetRootRegion());

    tmp = trav.next();
    assert(tmp == n1);
    tmp = trav.next();
    assert(tmp == n2);
    tmp = trav.next();
    assert(tmp == n3);
    tmp = trav.next();
    assert(tmp == nullptr);
  }
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TopdownTraverserTest-testOrderEnforcement", testOrderEnforcement);

static void
testInsertion()
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
    const jlm::rvsdg::Node * node = nullptr;
    jlm::rvsdg::TopDownTraverser trav(&graph.GetRootRegion());

    node = trav.next();
    assert(node == n1);

    /* At this point, n1 has been visited, now create some nodes */

    auto n3 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type });
    auto n4 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n3->output(0) }, {});
    auto n5 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n2->output(0) }, {});

    /*
      The newly created nodes n3 and n4 will not be visited,
      as they were not created as descendants of unvisited
      nodes. n5 must be visited, as n2 has not been visited yet.
    */

    std::cout << "start test\n";
    bool visited_n2 = false, visited_n3 = false, visited_n4 = false, visited_n5 = false;
    for (;;)
    {
      node = trav.next();
      std::cout << "visiting " << node << "\n";
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
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TopdownTraverserTest-testInsertion", testInsertion);

static void
testInsertingTopNode()
{
  // Starts with a graph like
  // n1 -> n2 -> GraphExport
  //
  // when n1 is visited, the graph is converted into
  //
  // n1
  // nX -> n2 -> GraphExport
  //
  // Since nX is created without any unvisited predecessors, it should not be visited.
  // n2 should be visited, however.

  jlm::rvsdg::Graph graph;
  auto type = jlm::tests::ValueType::Create();

  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type });
  auto n2 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1->output(0) }, { type });

  jlm::rvsdg::GraphExport::Create(*n2->output(0), "dummy");

  // Act and assert
  jlm::rvsdg::TopDownTraverser trav(&graph.GetRootRegion());
  auto node = trav.next();
  assert(node == n1);

  auto nX = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { }, { type });
  n1->output(0)->divert_users(nX->output(0));

  node = trav.next();
  assert(node == n2);

  node = trav.next();
  assert(node == nullptr);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TopdownTraverserTest-testInsertingTopNode", testInsertingTopNode);

static void
testMutating()
{
  auto test = [](jlm::rvsdg::Graph * graph,
                 jlm::rvsdg::Node * n1,
                 jlm::rvsdg::Node * n2,
                 jlm::rvsdg::Node * n3)
  {
    bool seen_n1 = false;
    bool seen_n2 = false;
    bool seen_n3 = false;

    for (const auto & tmp : jlm::rvsdg::TopDownTraverser(&graph->GetRootRegion()))
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

  jlm::rvsdg::Graph graph;
  auto type = jlm::tests::ValueType::Create();
  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type });
  auto n2 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type });
  auto n3 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1->output(0) }, {});

  test(&graph, n1, n2, n3);
  test(&graph, n1, n2, n3);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TopdownTraverserTest-testMutating", testMutating);

static void
testReplacement()
{
  // Starts with a graph like
  // n1 -> n2 -> n3 -> n4 -> GraphExport
  //         \-> n5 -> GraphExport
  //
  // when n2 is visited, the graph is converted into
  //
  // n1 -> n2 -> n3
  //   \-> nX -> nY -> n4 -> GraphExport
  //         \-> n5 -> GraphExport
  //
  // Since nX is created as a successor of only visited nodes, it is not visited.
  // nY is not visited either, as its predecessor nX is already "visited".
  // n3, n4 and n5 should be visited, however.

  jlm::rvsdg::Graph graph;
  auto type = jlm::tests::ValueType::Create();

  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { type });
  auto n2 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1->output(0) }, { type });
  auto n3 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n2->output(0) }, { type });
  auto n4 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n3->output(0) }, { type });
  auto n5 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n2->output(0) }, { type });

  jlm::rvsdg::GraphExport::Create(*n4->output(0), "dummy");

  // Act and assert
  jlm::rvsdg::TopDownTraverser trav(&graph.GetRootRegion());
  auto node = trav.next();
  assert(node == n1);

  node = trav.next();
  assert(node == n2);
  auto nX = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1->output(0) }, { type });
  auto nY = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { nX->output(0) }, { type });
  n3->output(0)->divert_users(nY->output(0));
  n5->input(0)->divert_to(nX->output(0));

  auto next1 = trav.next();
  auto next2 = trav.next();
  auto next3 = trav.next();
  assert(n3 == next1 || n3 == next2 || n3 == next3);
  assert(n4 == next1 || n4 == next2 || n4 == next3);
  assert(n5 == next1 || n5 == next2 || n5 == next3);

  auto next4 = trav.next();
  assert(next4 == nullptr);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TopdownTraverserTest-testReplacement", testReplacement);
