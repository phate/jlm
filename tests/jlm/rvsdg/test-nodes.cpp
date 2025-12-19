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

static void
RemoveOutputs()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  const auto valueType = ValueType::Create();

  Graph rvsdg;
  auto node = TestOperation::create(
      &rvsdg.GetRootRegion(),
      {},
      std::vector<std::shared_ptr<const Type>>{ valueType,
                                                valueType,
                                                valueType,
                                                valueType,
                                                valueType,
                                                valueType,
                                                valueType,
                                                valueType,
                                                valueType,
                                                valueType });

  auto & x1 = GraphExport::Create(*node->output(1), "x1");
  auto & x2 = GraphExport::Create(*node->output(2), "x2");
  auto & x3 = GraphExport::Create(*node->output(3), "x3");
  auto & x4 = GraphExport::Create(*node->output(4), "x4");
  auto & x5 = GraphExport::Create(*node->output(5), "x5");
  auto & x6 = GraphExport::Create(*node->output(6), "x6");
  auto & x7 = GraphExport::Create(*node->output(7), "x7");
  auto & x9 = GraphExport::Create(*node->output(9), "x9");

  // Act & Assert
  // Remove all outputs that have an even index
  size_t numRemovedOutputs = node->RemoveOutputs({ 0, 2, 4, 6, 8 });
  // We expect only output0 and output8 to be removed, as output2, output4, and
  // output6 are not dead
  assert(numRemovedOutputs == 2);
  assert(node->noutputs() == 8);
  assert(x1.origin()->index() == 0);
  assert(x2.origin()->index() == 1);
  assert(x3.origin()->index() == 2);
  assert(x4.origin()->index() == 3);
  assert(x5.origin()->index() == 4);
  assert(x6.origin()->index() == 5);
  assert(x7.origin()->index() == 6);
  assert(x9.origin()->index() == 7);

  // Remove all users from outputs
  rvsdg.GetRootRegion().RemoveResult(7);
  rvsdg.GetRootRegion().RemoveResult(6);
  rvsdg.GetRootRegion().RemoveResult(5);
  rvsdg.GetRootRegion().RemoveResult(4);
  rvsdg.GetRootRegion().RemoveResult(3);
  rvsdg.GetRootRegion().RemoveResult(2);
  rvsdg.GetRootRegion().RemoveResult(1);
  rvsdg.GetRootRegion().RemoveResult(0);

  // Remove all outputs that have an even index
  numRemovedOutputs = node->RemoveOutputs({ 0, 2, 4, 6 });
  assert(numRemovedOutputs == 4);
  assert(node->noutputs() == 4);
  assert(node->output(0)->index() == 0);
  assert(node->output(1)->index() == 1);
  assert(node->output(2)->index() == 2);
  assert(node->output(3)->index() == 3);

  // Remove no output
  numRemovedOutputs = node->RemoveOutputs({});
  assert(numRemovedOutputs == 0);
  assert(node->noutputs() == 4);
  assert(node->output(0)->index() == 0);
  assert(node->output(1)->index() == 1);
  assert(node->output(2)->index() == 2);
  assert(node->output(3)->index() == 3);

  // Remove non-existent output
  numRemovedOutputs = node->RemoveOutputs({ 15 });
  assert(numRemovedOutputs == 0);
  assert(node->noutputs() == 4);

  // Remove all remaining arguments
  numRemovedOutputs = node->RemoveOutputs({ 0, 1, 2, 3 });
  assert(numRemovedOutputs == 4);
  assert(node->noutputs() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes-RemoveOutputs", RemoveOutputs)

static void
RemoveInputs()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  Graph rvsdg;
  const RecordingObserver observer(rvsdg.GetRootRegion());
  const auto valueType = ValueType::Create();

  auto i0 = &GraphImport::Create(rvsdg, valueType, "i0");
  auto i1 = &GraphImport::Create(rvsdg, valueType, "i1");
  auto i2 = &GraphImport::Create(rvsdg, valueType, "i2");
  auto i3 = &GraphImport::Create(rvsdg, valueType, "i3");
  auto i4 = &GraphImport::Create(rvsdg, valueType, "i4");
  auto i5 = &GraphImport::Create(rvsdg, valueType, "i5");
  auto i6 = &GraphImport::Create(rvsdg, valueType, "i6");
  auto i7 = &GraphImport::Create(rvsdg, valueType, "i7");
  auto i8 = &GraphImport::Create(rvsdg, valueType, "i8");
  auto i9 = &GraphImport::Create(rvsdg, valueType, "i9");

  auto node =
      TestOperation::create(&rvsdg.GetRootRegion(), { i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 }, {});

  // Act & Assert
  assert(rvsdg.GetRootRegion().numTopNodes() == 0);

  // Remove all inputs that have an even index
  size_t numRemovedInputs = node->RemoveInputs({ 0, 2, 4, 6, 8 }, true);
  assert(numRemovedInputs == 5);
  assert(node->ninputs() == 5);
  assert(node->input(0)->origin() == i1);
  assert(node->input(1)->origin() == i3);
  assert(node->input(2)->origin() == i5);
  assert(node->input(3)->origin() == i7);
  assert(node->input(4)->origin() == i9);
  assert(i0->nusers() == 0);
  assert(i2->nusers() == 0);
  assert(i4->nusers() == 0);
  assert(i6->nusers() == 0);
  assert(i8->nusers() == 0);
  // We specified that the region is notified about the input removal
  assert(observer.destroyedInputIndices() == std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove no input
  numRemovedInputs = node->RemoveInputs({}, true);
  assert(numRemovedInputs == 0);
  assert(node->ninputs() == 5);
  assert(observer.destroyedInputIndices() == std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove non-existent input
  numRemovedInputs = node->RemoveInputs({ 15 }, true);
  assert(numRemovedInputs == 0);
  assert(node->ninputs() == 5);
  assert(observer.destroyedInputIndices() == std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove remaining inputs
  numRemovedInputs = node->RemoveInputs({ 0, 1, 2, 3, 4 }, false);
  assert(numRemovedInputs == 5);
  assert(node->ninputs() == 0);
  assert(i1->nusers() == 0);
  assert(i3->nusers() == 0);
  assert(i5->nusers() == 0);
  assert(i7->nusers() == 0);
  assert(i9->nusers() == 0);
  // We specified that the region is not notified about the input removal
  assert(observer.destroyedInputIndices() == std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Check that node is a top node
  assert(rvsdg.GetRootRegion().numTopNodes() == 1);
  assert(&*rvsdg.GetRootRegion().TopNodes().begin() == node);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes-RemoveInputs", RemoveInputs)

static void
test_nodes()
{
  test_node_copy();
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
