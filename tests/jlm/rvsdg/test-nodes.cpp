/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/HashSet.hpp>

TEST(NodeTests, test_node_copy)
{
  using namespace jlm::rvsdg;

  auto stateType = TestType::createStateType();
  auto valueType = TestType::createValueType();

  Graph graph;
  auto & s = GraphImport::Create(graph, stateType, "");
  auto & v = GraphImport::Create(graph, valueType, "");

  auto structuralNode1 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto inputVar11 = structuralNode1->addInputWithArguments(s);
  auto inputVar12 = structuralNode1->addInputWithArguments(s);
  auto & output11 = structuralNode1->addOutputOnly(stateType);
  auto & output12 = structuralNode1->addOutputOnly(valueType);

  auto simpleNode1 = TestOperation::createNode(
      structuralNode1->subregion(0),
      { inputVar11.argument[0] },
      { stateType });
  auto simpleNode2 = TestOperation::createNode(
      structuralNode1->subregion(0),
      { inputVar12.argument[0] },
      { valueType });

  RegionResult::Create(
      *structuralNode1->subregion(0),
      *simpleNode1->output(0),
      &output11,
      stateType);
  RegionResult::Create(
      *structuralNode1->subregion(0),
      *simpleNode2->output(0),
      &output12,
      valueType);

  view(&graph.GetRootRegion(), stdout);

  // Act & Assert
  // copy with arguments and results
  {
    auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
    auto & input21 = structuralNode2->addInputOnly(s);
    auto & input22 = structuralNode2->addInputOnly(s);
    auto & output21 = structuralNode2->addOutputOnly(stateType);
    auto & output22 = structuralNode2->addOutputOnly(valueType);

    SubstitutionMap smap;
    smap.insert(inputVar11.input, &input21);
    smap.insert(inputVar12.input, &input22);
    smap.insert(&output11, &output21);
    smap.insert(&output12, &output22);

    structuralNode1->subregion(0)->copy(structuralNode2->subregion(0), smap, true, true);
    view(&graph.GetRootRegion(), stdout);

    auto subregion = structuralNode2->subregion(0);
    EXPECT_EQ(subregion->narguments(), 2);
    EXPECT_EQ(subregion->argument(0)->input(), &input21);
    EXPECT_EQ(subregion->argument(1)->input(), &input22);

    EXPECT_EQ(subregion->nresults(), 2);
    EXPECT_EQ(subregion->result(0)->output(), &output21);
    EXPECT_EQ(subregion->result(1)->output(), &output22);

    EXPECT_EQ(subregion->numNodes(), 2);
  }

  // copy without arguments
  {
    auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
    auto inputVar21 = structuralNode2->addArguments(stateType);
    auto inputVar22 = structuralNode2->addArguments(stateType);
    auto & output21 = structuralNode2->addOutputOnly(stateType);
    auto & output22 = structuralNode2->addOutputOnly(valueType);

    SubstitutionMap smap2;
    smap2.insert(structuralNode1->subregion(0)->argument(0), inputVar21.argument[0]);
    smap2.insert(structuralNode1->subregion(0)->argument(1), inputVar22.argument[0]);
    smap2.insert(&output11, &output21);
    smap2.insert(&output12, &output22);

    structuralNode1->subregion(0)->copy(structuralNode2->subregion(0), smap2, false, true);
    view(&graph.GetRootRegion(), stdout);

    auto subregion = structuralNode2->subregion(0);
    EXPECT_EQ(subregion->nresults(), 2);
    EXPECT_EQ(subregion->result(0)->output(), &output21);
    EXPECT_EQ(subregion->result(1)->output(), &output22);

    EXPECT_EQ(subregion->numNodes(), 2);
  }

  // copy structural node
  {
    SubstitutionMap smap3;
    smap3.insert(&s, &s);
    smap3.insert(&v, &v);

    structuralNode1->copy(&graph.GetRootRegion(), smap3);
    view(&graph.GetRootRegion(), stdout);

    EXPECT_EQ(graph.GetRootRegion().numNodes(), 4);
  }
}

TEST(NodeTests, RemoveOutputs)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto node = TestOperation::createNode(
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
  EXPECT_EQ(numRemovedOutputs, 2);
  EXPECT_EQ(node->noutputs(), 8);
  EXPECT_EQ(x1.origin()->index(), 0);
  EXPECT_EQ(x2.origin()->index(), 1);
  EXPECT_EQ(x3.origin()->index(), 2);
  EXPECT_EQ(x4.origin()->index(), 3);
  EXPECT_EQ(x5.origin()->index(), 4);
  EXPECT_EQ(x6.origin()->index(), 5);
  EXPECT_EQ(x7.origin()->index(), 6);
  EXPECT_EQ(x9.origin()->index(), 7);

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
  EXPECT_EQ(numRemovedOutputs, 4);
  EXPECT_EQ(node->noutputs(), 4);
  EXPECT_EQ(node->output(0)->index(), 0);
  EXPECT_EQ(node->output(1)->index(), 1);
  EXPECT_EQ(node->output(2)->index(), 2);
  EXPECT_EQ(node->output(3)->index(), 3);

  // Remove no output
  numRemovedOutputs = node->RemoveOutputs({});
  EXPECT_EQ(numRemovedOutputs, 0);
  EXPECT_EQ(node->noutputs(), 4);
  EXPECT_EQ(node->output(0)->index(), 0);
  EXPECT_EQ(node->output(1)->index(), 1);
  EXPECT_EQ(node->output(2)->index(), 2);
  EXPECT_EQ(node->output(3)->index(), 3);

  // Remove non-existent output
  numRemovedOutputs = node->RemoveOutputs({ 15 });
  EXPECT_EQ(numRemovedOutputs, 0);
  EXPECT_EQ(node->noutputs(), 4);

  // Remove all remaining outputs
  numRemovedOutputs = node->RemoveOutputs({ 0, 1, 2, 3 });
  EXPECT_EQ(numRemovedOutputs, 4);
  EXPECT_EQ(node->noutputs(), 0);
}

TEST(NodeTests, RemoveInputs)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  const RecordingObserver observer(rvsdg.GetRootRegion());
  const auto valueType = TestType::createValueType();

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

  auto node = TestOperation::createNode(
      &rvsdg.GetRootRegion(),
      { i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 },
      {});

  // Act & Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numTopNodes(), 0);

  // Remove all inputs that have an even index
  size_t numRemovedInputs = node->RemoveInputs({ 0, 2, 4, 6, 8 }, true);
  EXPECT_EQ(numRemovedInputs, 5);
  EXPECT_EQ(node->ninputs(), 5);
  EXPECT_EQ(node->input(0)->origin(), i1);
  EXPECT_EQ(node->input(1)->origin(), i3);
  EXPECT_EQ(node->input(2)->origin(), i5);
  EXPECT_EQ(node->input(3)->origin(), i7);
  EXPECT_EQ(node->input(4)->origin(), i9);
  EXPECT_EQ(i0->nusers(), 0);
  EXPECT_EQ(i2->nusers(), 0);
  EXPECT_EQ(i4->nusers(), 0);
  EXPECT_EQ(i6->nusers(), 0);
  EXPECT_EQ(i8->nusers(), 0);
  // We specified that the region is notified about the input removal
  EXPECT_EQ(observer.destroyedInputIndices(), std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove no input
  numRemovedInputs = node->RemoveInputs({}, true);
  EXPECT_EQ(numRemovedInputs, 0);
  EXPECT_EQ(node->ninputs(), 5);
  EXPECT_EQ(observer.destroyedInputIndices(), std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove non-existent input
  numRemovedInputs = node->RemoveInputs({ 15 }, true);
  EXPECT_EQ(numRemovedInputs, 0);
  EXPECT_EQ(node->ninputs(), 5);
  EXPECT_EQ(observer.destroyedInputIndices(), std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove remaining inputs
  numRemovedInputs = node->RemoveInputs({ 0, 1, 2, 3, 4 }, false);
  EXPECT_EQ(numRemovedInputs, 5);
  EXPECT_EQ(node->ninputs(), 0);
  EXPECT_EQ(i1->nusers(), 0);
  EXPECT_EQ(i3->nusers(), 0);
  EXPECT_EQ(i5->nusers(), 0);
  EXPECT_EQ(i7->nusers(), 0);
  EXPECT_EQ(i9->nusers(), 0);
  // We specified that the region is not notified about the input removal
  EXPECT_EQ(observer.destroyedInputIndices(), std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Check that node is a top node
  EXPECT_EQ(rvsdg.GetRootRegion().numTopNodes(), 1);
  EXPECT_EQ(&*rvsdg.GetRootRegion().TopNodes().begin(), node);
}

TEST(NodeTests, NodeInputIteration)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto i = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<TestOperation>(
      { i, i, i, i, i },
      std::vector<std::shared_ptr<const Type>>(5, valueType),
      std::vector<std::shared_ptr<const Type>>{ valueType });

  GraphExport::Create(*node.output(0), "x0");

  // Act & Assert
  size_t n = 0;
  for (auto & input : node.Inputs())
  {
    EXPECT_EQ(&input, node.input(n++));
  }
  EXPECT_EQ(n, node.ninputs());

  n = 0;
  const Node * constNode = &node;
  for (auto & input : constNode->Inputs())
  {
    EXPECT_EQ(&input, node.input(n++));
  }
  EXPECT_EQ(n, node.ninputs());
}

TEST(NodeTests, NodeOutputIteration)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto i = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<TestOperation>(
      { i },
      std::vector<std::shared_ptr<const Type>>{ valueType },
      std::vector<std::shared_ptr<const Type>>(5, valueType));

  GraphExport::Create(*node.output(0), "x0");

  // Act & Assert
  size_t n = 0;
  for (auto & output : node.Outputs())
  {
    EXPECT_EQ(&output, node.output(n++));
  }
  EXPECT_EQ(n, node.noutputs());

  n = 0;
  const Node * constNode = &node;
  for (auto & output : constNode->Outputs())
  {
    EXPECT_EQ(&output, constNode->output(n++));
  }
  EXPECT_EQ(n, constNode->noutputs());
}

TEST(NodeTests, zeroInputOutputIteration)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto node = TestOperation::createNode(&rvsdg.GetRootRegion(), {}, {});

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

  EXPECT_FALSE(enteredLoopBody);
}

TEST(NodeTests, NodeId)
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange & Act & Assert
  Graph rvsdg1;
  HashSet<Node::Id> NodeIds;

  auto node0 = TestOperation::createNode(&rvsdg1.GetRootRegion(), {}, {});
  auto node1 = TestOperation::createNode(&rvsdg1.GetRootRegion(), {}, {});
  auto node2 = TestOperation::createNode(&rvsdg1.GetRootRegion(), {}, {});

  NodeIds.insert(node0->GetNodeId());
  NodeIds.insert(node1->GetNodeId());
  NodeIds.insert(node2->GetNodeId());

  // We should have three unique identifiers in the set
  EXPECT_EQ(NodeIds.Size(), 3);

  // The identifiers should be consecutive as no other nodes where created in between those
  // three nodes
  EXPECT_EQ(node0->GetNodeId(), 0);
  EXPECT_EQ(node1->GetNodeId(), 1);
  EXPECT_EQ(node2->GetNodeId(), 2);

  // Removing a node should not change the identifiers of the other nodes
  remove(node1);
  EXPECT_EQ(node0->GetNodeId(), 0);
  EXPECT_EQ(node2->GetNodeId(), 2);

  // Adding a new node should give us the next identifier as no other nodes have been created in
  // between
  auto node3 = TestOperation::createNode(&rvsdg1.GetRootRegion(), {}, {});
  EXPECT_EQ(node3->GetNodeId(), 3);

  // Identifiers should be only unique for each region
  Graph rvsdg2;
  auto node4 = TestOperation::createNode(&rvsdg2.GetRootRegion(), {}, {});
  EXPECT_EQ(node4->GetNodeId(), 0);
}
