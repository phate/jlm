/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/HashSet.hpp>

static void
test_node_copy()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto stype = jlm::tests::StateType::Create();
  auto vtype = jlm::tests::ValueType::Create();

  Graph graph;
  auto & s = jlm::rvsdg::GraphImport::Create(graph, stype, "");
  auto & v = jlm::rvsdg::GraphImport::Create(graph, vtype, "");

  auto n1 = TestStructuralNode::create(&graph.GetRootRegion(), 3);
  auto & i1 = n1->addInputOnly(s);
  auto & i2 = n1->addInputOnly(v);
  auto & o1 = n1->addOutputOnly(stype);
  auto & o2 = n1->addOutputOnly(vtype);

  auto & a1 = TestGraphArgument::Create(*n1->subregion(0), &i1, stype);
  auto & a2 = TestGraphArgument::Create(*n1->subregion(0), &i2, vtype);

  auto n2 = TestOperation::create(n1->subregion(0), { &a1 }, { stype });
  auto n3 = TestOperation::create(n1->subregion(0), { &a2 }, { vtype });

  RegionResult::Create(*n1->subregion(0), *n2->output(0), &o1, stype);
  RegionResult::Create(*n1->subregion(0), *n3->output(0), &o2, vtype);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  /* copy first into second region with arguments and results */
  SubstitutionMap smap;
  smap.insert(&i1, &i1);
  smap.insert(&i2, &i2);
  smap.insert(&o1, &o1);
  smap.insert(&o2, &o2);
  n1->subregion(0)->copy(n1->subregion(1), smap, true, true);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  auto r2 = n1->subregion(1);
  assert(r2->narguments() == 2);
  assert(r2->argument(0)->input() == &i1);
  assert(r2->argument(1)->input() == &i2);

  assert(r2->nresults() == 2);
  assert(r2->result(0)->output() == &o1);
  assert(r2->result(1)->output() == &o2);

  assert(r2->numNodes() == 2);

  /* copy second into third region only with arguments */
  jlm::rvsdg::SubstitutionMap smap2;
  auto & a3 = TestGraphArgument::Create(*n1->subregion(2), &i1, stype);
  auto & a4 = TestGraphArgument::Create(*n1->subregion(2), &i2, vtype);
  smap2.insert(r2->argument(0), &a3);
  smap2.insert(r2->argument(1), &a4);

  smap2.insert(&o1, &o1);
  smap2.insert(&o2, &o2);
  n1->subregion(1)->copy(n1->subregion(2), smap2, false, true);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  auto r3 = n1->subregion(2);
  assert(r3->nresults() == 2);
  assert(r3->result(0)->output() == &o1);
  assert(r3->result(1)->output() == &o2);

  assert(r3->numNodes() == 2);

  /* copy structural node */
  jlm::rvsdg::SubstitutionMap smap3;
  smap3.insert(&s, &s);
  smap3.insert(&v, &v);
  n1->copy(&graph.GetRootRegion(), smap3);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  assert(graph.GetRootRegion().numNodes() == 2);
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
  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");

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
  auto i = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      { i, i, i, i, i },
      std::vector<std::shared_ptr<const Type>>(5, valueType),
      std::vector<std::shared_ptr<const Type>>{ valueType });

  GraphExport::Create(*node.output(0), "x0");

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
  auto i = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      { i },
      std::vector<std::shared_ptr<const Type>>{ valueType },
      std::vector<std::shared_ptr<const Type>>(5, valueType));

  GraphExport::Create(*node.output(0), "x0");

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

static void
zeroInputOutputIteration()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  Graph rvsdg;
  auto node = TestOperation::create(&rvsdg.GetRootRegion(), {}, {});

  // Act & Assert
  bool enteredLoopBody = false;
  for ([[maybe_unused]] auto & _ : node->Inputs())
  {
    enteredLoopBody = true;
  }
  for ([[maybe_unused]] auto & _ : node->Outputs())
  {
    enteredLoopBody = true;
  }

  const Node * constNode = node;
  for ([[maybe_unused]] auto & _ : constNode->Inputs())
  {
    enteredLoopBody = true;
  }
  for ([[maybe_unused]] auto & _ : constNode->Outputs())
  {
    enteredLoopBody = true;
  }

  assert(enteredLoopBody == false);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes-zeroInputOutputIteration", zeroInputOutputIteration)

static void
NodeId()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange & Act & Assert
  Graph rvsdg1;
  HashSet<Node::Id> NodeIds;

  auto node0 = TestOperation::create(&rvsdg1.GetRootRegion(), {}, {});
  auto node1 = TestOperation::create(&rvsdg1.GetRootRegion(), {}, {});
  auto node2 = TestOperation::create(&rvsdg1.GetRootRegion(), {}, {});

  NodeIds.insert(node0->GetNodeId());
  NodeIds.insert(node1->GetNodeId());
  NodeIds.insert(node2->GetNodeId());

  // We should have three unique identifiers in the set
  assert(NodeIds.Size() == 3);

  // The identifiers should be consecutive as no other nodes where created in between those
  // three nodes
  assert(node0->GetNodeId() == 0);
  assert(node1->GetNodeId() == 1);
  assert(node2->GetNodeId() == 2);

  // Removing a node should not change the identifiers of the other nodes
  remove(node1);
  assert(node0->GetNodeId() == 0);
  assert(node2->GetNodeId() == 2);

  // Adding a new node should give us the next identifier as no other nodes have been created in
  // between
  auto node3 = TestOperation::create(&rvsdg1.GetRootRegion(), {}, {});
  assert(node3->GetNodeId() == 3);

  // Identifiers should be only unique for each region
  Graph rvsdg2;
  auto node4 = TestOperation::create(&rvsdg2.GetRootRegion(), {}, {});
  assert(node4->GetNodeId() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes-NodeId", NodeId)

static void
TraceOutputIntraProcedural_Gamma()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Assert
  const auto controlType = ControlType::Create(2);
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, controlType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i2 = GraphImport::Create(rvsdg, valueType, "i2");

  const auto gammaNode = GammaNode::create(&i0, 2);
  auto entryVar1 = gammaNode->AddEntryVar(&i1);
  auto entryVar2 = gammaNode->AddEntryVar(&i2);

  auto node = TestOperation::create(
      gammaNode->subregion(1),
      { entryVar2.branchArgument[1] },
      { valueType });

  auto exitVar1 =
      gammaNode->AddExitVar({ entryVar1.branchArgument[0], entryVar1.branchArgument[1] });
  auto exitVar2 = gammaNode->AddExitVar({ entryVar2.branchArgument[0], node->output(0) });

  auto & x0 = GraphExport::Create(*exitVar1.output, "x0");
  auto & x1 = GraphExport::Create(*exitVar2.output, "x1");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin());
  const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin());
  const auto & traceGammaEntry = traceOutputIntraProcedurally(*entryVar1.branchArgument[0]);
  const auto & tracedNodeInput = traceOutputIntraProcedurally(*node->input(0)->origin());

  // Assert
  assert(&tracedX0 == &i1);
  assert(&tracedX1 == x1.origin());
  assert(&traceGammaEntry == &i1);
  assert(&tracedNodeInput == &i2);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-nodes-TraceOutputIntraProcedural_Gamma",
    TraceOutputIntraProcedural_Gamma)

static void
TraceOutputIntraProcedural_Theta()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Assert
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i2");

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVar0 = thetaNode->AddLoopVar(&i0);
  auto loopVar1 = thetaNode->AddLoopVar(&i1);

  auto node = TestOperation::create(thetaNode->subregion(), { loopVar1.pre }, { valueType });
  loopVar1.post->divert_to(node->output(0));

  auto & x0 = GraphExport::Create(*loopVar0.output, "x0");
  auto & x1 = GraphExport::Create(*loopVar1.output, "x1");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin());
  const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin());
  const auto & traceGammaEntry = traceOutputIntraProcedurally(*loopVar0.pre);
  const auto & tracedNodeInput = traceOutputIntraProcedurally(*node->input(0)->origin());

  // Assert
  assert(&tracedX0 == &i0);
  assert(&tracedX1 == x1.origin());
  assert(&traceGammaEntry == &i0);
  assert(&tracedNodeInput == loopVar1.pre);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-nodes-TraceOutputIntraProcedural_Theta",
    TraceOutputIntraProcedural_Theta)
