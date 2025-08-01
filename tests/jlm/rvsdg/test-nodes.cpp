/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

static void
test_node_copy()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto stype = jlm::tests::StateType::Create();
  auto vtype = jlm::tests::ValueType::Create();

  Graph graph;
  auto s = &jlm::tests::GraphImport::Create(graph, stype, "");
  auto v = &jlm::tests::GraphImport::Create(graph, vtype, "");

  auto n1 = TestStructuralNode::create(&graph.GetRootRegion(), 3);
  auto i1 = StructuralInput::create(n1, s, stype);
  auto i2 = StructuralInput::create(n1, v, vtype);
  auto o1 = StructuralOutput::create(n1, stype);
  auto o2 = StructuralOutput::create(n1, vtype);

  auto & a1 = TestGraphArgument::Create(*n1->subregion(0), i1, stype);
  auto & a2 = TestGraphArgument::Create(*n1->subregion(0), i2, vtype);

  auto n2 = TestOperation::create(n1->subregion(0), { &a1 }, { stype });
  auto n3 = TestOperation::create(n1->subregion(0), { &a2 }, { vtype });

  RegionResult::Create(*n1->subregion(0), *n2->output(0), o1, stype);
  RegionResult::Create(*n1->subregion(0), *n3->output(0), o2, vtype);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  /* copy first into second region with arguments and results */
  SubstitutionMap smap;
  smap.insert(i1, i1);
  smap.insert(i2, i2);
  smap.insert(o1, o1);
  smap.insert(o2, o2);
  n1->subregion(0)->copy(n1->subregion(1), smap, true, true);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  auto r2 = n1->subregion(1);
  assert(r2->narguments() == 2);
  assert(r2->argument(0)->input() == i1);
  assert(r2->argument(1)->input() == i2);

  assert(r2->nresults() == 2);
  assert(r2->result(0)->output() == o1);
  assert(r2->result(1)->output() == o2);

  assert(r2->nnodes() == 2);

  /* copy second into third region only with arguments */
  jlm::rvsdg::SubstitutionMap smap2;
  auto & a3 = TestGraphArgument::Create(*n1->subregion(2), i1, stype);
  auto & a4 = TestGraphArgument::Create(*n1->subregion(2), i2, vtype);
  smap2.insert(r2->argument(0), &a3);
  smap2.insert(r2->argument(1), &a4);

  smap2.insert(o1, o1);
  smap2.insert(o2, o2);
  n1->subregion(1)->copy(n1->subregion(2), smap2, false, true);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  auto r3 = n1->subregion(2);
  assert(r3->nresults() == 2);
  assert(r3->result(0)->output() == o1);
  assert(r3->result(1)->output() == o2);

  assert(r3->nnodes() == 2);

  /* copy structural node */
  jlm::rvsdg::SubstitutionMap smap3;
  smap3.insert(s, s);
  smap3.insert(v, v);
  n1->copy(&graph.GetRootRegion(), smap3);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  assert(graph.GetRootRegion().nnodes() == 2);
}

static inline void
test_node_depth()
{
  auto vt = jlm::tests::ValueType::Create();

  jlm::rvsdg::Graph graph;
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto null = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { vt });
  auto bin =
      jlm::tests::TestOperation::create(&graph.GetRootRegion(), { null->output(0), x }, { vt });
  auto un = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { bin->output(0) }, { vt });

  jlm::tests::GraphExport::Create(*un->output(0), "x");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  assert(null->depth() == 0);
  assert(bin->depth() == 1);
  assert(un->depth() == 2);

  bin->input(0)->divert_to(x);
  assert(bin->depth() == 0);
  assert(un->depth() == 1);
}

/**
 * Test Node::RemoveOutputsWhere()
 */
static void
TestRemoveOutputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;

  auto valueType = jlm::tests::ValueType::Create();
  auto & node1 = CreateOpNode<jlm::tests::TestOperation>(
      rvsdg.GetRootRegion(),
      std::vector<std::shared_ptr<const Type>>(),
      std::vector<std::shared_ptr<const Type>>{ valueType, valueType, valueType });
  auto output0 = node1.output(0);
  auto output2 = node1.output(2);

  auto & node2 = CreateOpNode<jlm::tests::TestOperation>(
      std::vector<Output *>({ output0, output2 }),
      std::vector<std::shared_ptr<const Type>>{ valueType, valueType },
      std::vector<std::shared_ptr<const Type>>{ valueType, valueType });

  // Act & Assert
  node2.RemoveOutputsWhere(
      [](const jlm::rvsdg::Output &)
      {
        return false;
      });
  assert(node2.noutputs() == 2);

  node1.RemoveOutputsWhere(
      [](const jlm::rvsdg::Output &)
      {
        return true;
      });
  assert(node1.noutputs() == 2);
  assert(node1.output(0) == output0);
  assert(node1.output(0)->index() == 0);
  assert(node1.output(1) == output2);
  assert(node1.output(1)->index() == 1);

  node2.RemoveOutputsWhere(
      [](const jlm::rvsdg::Output &)
      {
        return true;
      });
  assert(node2.noutputs() == 0);

  remove(&node2);

  node1.RemoveOutputsWhere(
      [](const jlm::rvsdg::Output & output)
      {
        return output.index() == 0;
      });
  assert(node1.noutputs() == 1);
  assert(node1.output(0) == output2);
  assert(node1.output(0)->index() == 0);

  node1.RemoveOutputsWhere(
      [](const jlm::rvsdg::Output &)
      {
        return true;
      });
  assert(node1.noutputs() == 0);
}

/**
 * Test Node::RemoveInputsWhere()
 */
static void
TestRemoveInputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  jlm::rvsdg::Graph rvsdg;
  auto valueType = jlm::tests::ValueType::Create();
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      { x, x, x },
      std::vector<std::shared_ptr<const Type>>{ valueType, valueType, valueType },
      std::vector<std::shared_ptr<const Type>>{});
  auto input0 = node.input(0);
  auto input2 = node.input(2);

  // Act & Assert
  node.RemoveInputsWhere(
      [](const jlm::rvsdg::Input & input)
      {
        return input.index() == 1;
      });
  assert(node.ninputs() == 2);
  assert(node.input(0) == input0);
  assert(node.input(1) == input2);

  node.RemoveInputsWhere(
      [](const jlm::rvsdg::Input &)
      {
        return true;
      });
  assert(node.ninputs() == 0);
}

static void
test_nodes()
{
  test_node_copy();
  test_node_depth();
  TestRemoveOutputsWhere();
  TestRemoveInputsWhere();
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes", test_nodes)

static void
NodeInputIteration()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto i = &jlm::tests::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      { i, i, i, i, i },
      std::vector<std::shared_ptr<const Type>>(5, valueType),
      std::vector<std::shared_ptr<const Type>>{ valueType });

  jlm::tests::GraphExport::Create(*node.output(0), "x0");

  // Act & Assert
  size_t n = 0;
  for (auto & input : node.Inputs())
  {
    assert(&input == node.input(n++));
  }
  assert(n == node.ninputs());

  n = 0;
  const Node * constNode = &node;
  for (auto & input : constNode->Inputs())
  {
    assert(&input == node.input(n++));
  }
  assert(n == node.ninputs());
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes-NodeInputIteration", NodeInputIteration)

static void
NodeOutputIteration()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto i = &jlm::tests::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      { i },
      std::vector<std::shared_ptr<const Type>>{ valueType },
      std::vector<std::shared_ptr<const Type>>(5, valueType));

  jlm::tests::GraphExport::Create(*node.output(0), "x0");

  // Act & Assert
  size_t n = 0;
  for (auto & output : node.Outputs())
  {
    assert(&output == node.output(n++));
  }
  assert(n == node.noutputs());

  n = 0;
  const Node * constNode = &node;
  for (auto & output : constNode->Outputs())
  {
    assert(&output == constNode->output(n++));
  }
  assert(n == constNode->noutputs());
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes-NodeOutputIteration", NodeOutputIteration)
