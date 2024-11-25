/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

static void
test_flattened_binary_reduction()
{
  using namespace jlm::rvsdg;

  auto vt = jlm::tests::valuetype::Create();
  jlm::tests::binary_op op(vt, vt, binary_op::flags::associative);

  /* test paralell reduction */
  {
    Graph graph;
    auto i0 = &jlm::tests::GraphImport::Create(graph, vt, "");
    auto i1 = &jlm::tests::GraphImport::Create(graph, vt, "");
    auto i2 = &jlm::tests::GraphImport::Create(graph, vt, "");
    auto i3 = &jlm::tests::GraphImport::Create(graph, vt, "");

    auto o1 = simple_node::create_normalized(graph.root(), op, { i0, i1 })[0];
    auto o2 = simple_node::create_normalized(graph.root(), op, { o1, i2 })[0];
    auto o3 = simple_node::create_normalized(graph.root(), op, { o2, i3 })[0];

    auto & ex = jlm::tests::GraphExport::Create(*o3, "");
    graph.prune();

    jlm::rvsdg::view(graph, stdout);
    assert(
        graph.root()->nnodes() == 1 && Region::Contains<flattened_binary_op>(*graph.root(), false));

    flattened_binary_op::reduce(&graph, jlm::rvsdg::flattened_binary_op::reduction::parallel);
    jlm::rvsdg::view(graph, stdout);

    assert(graph.root()->nnodes() == 3);

    auto node0 = output::GetNode(*ex.origin());
    assert(is<jlm::tests::binary_op>(node0));

    auto node1 = output::GetNode(*node0->input(0)->origin());
    assert(is<jlm::tests::binary_op>(node1));

    auto node2 = output::GetNode(*node0->input(1)->origin());
    assert(is<jlm::tests::binary_op>(node2));
  }

  /* test linear reduction */
  {
    Graph graph;
    auto i0 = &jlm::tests::GraphImport::Create(graph, vt, "");
    auto i1 = &jlm::tests::GraphImport::Create(graph, vt, "");
    auto i2 = &jlm::tests::GraphImport::Create(graph, vt, "");
    auto i3 = &jlm::tests::GraphImport::Create(graph, vt, "");

    auto o1 = simple_node::create_normalized(graph.root(), op, { i0, i1 })[0];
    auto o2 = simple_node::create_normalized(graph.root(), op, { o1, i2 })[0];
    auto o3 = simple_node::create_normalized(graph.root(), op, { o2, i3 })[0];

    auto & ex = jlm::tests::GraphExport::Create(*o3, "");
    graph.prune();

    jlm::rvsdg::view(graph, stdout);
    assert(
        graph.root()->nnodes() == 1 && Region::Contains<flattened_binary_op>(*graph.root(), false));

    flattened_binary_op::reduce(&graph, jlm::rvsdg::flattened_binary_op::reduction::linear);
    jlm::rvsdg::view(graph, stdout);

    assert(graph.root()->nnodes() == 3);

    auto node0 = output::GetNode(*ex.origin());
    assert(is<jlm::tests::binary_op>(node0));

    auto node1 = output::GetNode(*node0->input(0)->origin());
    assert(is<jlm::tests::binary_op>(node1));

    auto node2 = output::GetNode(*node1->input(0)->origin());
    assert(is<jlm::tests::binary_op>(node2));
  }
}

static int
test_main()
{
  test_flattened_binary_reduction();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-binary", test_main)
