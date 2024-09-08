/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

static void
test_node_copy(void)
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto stype = jlm::tests::statetype::Create();
  auto vtype = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto s = &jlm::tests::GraphImport::Create(graph, stype, "");
  auto v = &jlm::tests::GraphImport::Create(graph, vtype, "");

  auto n1 = jlm::tests::structural_node::create(graph.root(), 3);
  auto i1 = structural_input::create(n1, s, stype);
  auto i2 = structural_input::create(n1, v, vtype);
  auto o1 = structural_output::create(n1, stype);
  auto o2 = structural_output::create(n1, vtype);

  auto & a1 = TestGraphArgument::Create(*n1->subregion(0), i1, stype);
  auto & a2 = TestGraphArgument::Create(*n1->subregion(0), i2, vtype);

  auto n2 = jlm::tests::test_op::create(n1->subregion(0), { &a1 }, { stype });
  auto n3 = jlm::tests::test_op::create(n1->subregion(0), { &a2 }, { vtype });

  TestGraphResult::Create(*n2->output(0), o1);
  TestGraphResult::Create(*n3->output(0), o2);

  jlm::rvsdg::view(graph.root(), stdout);

  /* copy first into second region with arguments and results */
  substitution_map smap;
  smap.insert(i1, i1);
  smap.insert(i2, i2);
  smap.insert(o1, o1);
  smap.insert(o2, o2);
  n1->subregion(0)->copy(n1->subregion(1), smap, true, true);

  jlm::rvsdg::view(graph.root(), stdout);

  auto r2 = n1->subregion(1);
  assert(r2->narguments() == 2);
  assert(r2->argument(0)->input() == i1);
  assert(r2->argument(1)->input() == i2);

  assert(r2->nresults() == 2);
  assert(r2->result(0)->output() == o1);
  assert(r2->result(1)->output() == o2);

  assert(r2->nnodes() == 2);

  /* copy second into third region only with arguments */
  jlm::rvsdg::substitution_map smap2;
  auto & a3 = TestGraphArgument::Create(*n1->subregion(2), i1, stype);
  auto & a4 = TestGraphArgument::Create(*n1->subregion(2), i2, vtype);
  smap2.insert(r2->argument(0), &a3);
  smap2.insert(r2->argument(1), &a4);

  smap2.insert(o1, o1);
  smap2.insert(o2, o2);
  n1->subregion(1)->copy(n1->subregion(2), smap2, false, true);

  jlm::rvsdg::view(graph.root(), stdout);

  auto r3 = n1->subregion(2);
  assert(r3->nresults() == 2);
  assert(r3->result(0)->output() == o1);
  assert(r3->result(1)->output() == o2);

  assert(r3->nnodes() == 2);

  /* copy structural node */
  jlm::rvsdg::substitution_map smap3;
  smap3.insert(s, s);
  smap3.insert(v, v);
  n1->copy(graph.root(), smap3);

  jlm::rvsdg::view(graph.root(), stdout);

  assert(graph.root()->nnodes() == 2);
}

static inline void
test_node_depth()
{
  auto vt = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto null = jlm::tests::test_op::create(graph.root(), {}, { vt });
  auto bin = jlm::tests::test_op::create(graph.root(), { null->output(0), x }, { vt });
  auto un = jlm::tests::test_op::create(graph.root(), { bin->output(0) }, { vt });

  jlm::tests::GraphExport::Create(*un->output(0), "x");

  jlm::rvsdg::view(graph.root(), stdout);

  assert(null->depth() == 0);
  assert(bin->depth() == 1);
  assert(un->depth() == 2);

  bin->input(0)->divert_to(x);
  assert(bin->depth() == 0);
  assert(un->depth() == 1);
}

/**
 * Test node::RemoveOutputsWhere()
 */
static void
TestRemoveOutputsWhere()
{
  // Arrange
  jlm::rvsdg::graph rvsdg;

  auto valueType = jlm::tests::valuetype::Create();
  auto & node1 =
      jlm::tests::SimpleNode::Create(*rvsdg.root(), {}, { valueType, valueType, valueType });
  auto output0 = node1.output(0);
  auto output2 = node1.output(2);

  auto & node2 =
      jlm::tests::SimpleNode::Create(*rvsdg.root(), { output0, output2 }, { valueType, valueType });

  // Act & Assert
  node2.RemoveOutputsWhere(
      [](const jlm::rvsdg::output & output)
      {
        return false;
      });
  assert(node2.noutputs() == 2);

  node1.RemoveOutputsWhere(
      [](const jlm::rvsdg::output & output)
      {
        return true;
      });
  assert(node1.noutputs() == 2);
  assert(node1.output(0) == output0);
  assert(node1.output(0)->index() == 0);
  assert(node1.output(1) == output2);
  assert(node1.output(1)->index() == 1);

  node2.RemoveOutputsWhere(
      [](const jlm::rvsdg::output & output)
      {
        return true;
      });
  assert(node2.noutputs() == 0);

  remove(&node2);

  node1.RemoveOutputsWhere(
      [](const jlm::rvsdg::output & output)
      {
        return output.index() == 0;
      });
  assert(node1.noutputs() == 1);
  assert(node1.output(0) == output2);
  assert(node1.output(0)->index() == 0);

  node1.RemoveOutputsWhere(
      [](const jlm::rvsdg::output & output)
      {
        return true;
      });
  assert(node1.noutputs() == 0);
}

/**
 * Test node::RemoveInputsWhere()
 */
static void
TestRemoveInputsWhere()
{
  // Arrange
  jlm::rvsdg::graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");

  auto & node = jlm::tests::SimpleNode::Create(*rvsdg.root(), { x, x, x }, {});
  auto input0 = node.input(0);
  auto input2 = node.input(2);

  // Act & Assert
  node.RemoveInputsWhere(
      [](const jlm::rvsdg::input & input)
      {
        return input.index() == 1;
      });
  assert(node.ninputs() == 2);
  assert(node.input(0) == input0);
  assert(node.input(1) == input2);

  node.RemoveInputsWhere(
      [](const jlm::rvsdg::input & input)
      {
        return true;
      });
  assert(node.ninputs() == 0);
}

static int
test_nodes()
{
  test_node_copy();
  test_node_depth();
  TestRemoveOutputsWhere();
  TestRemoveInputsWhere();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes", test_nodes)
