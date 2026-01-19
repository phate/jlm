/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(MemoryStateOperationTests, MemoryStateSplitEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  MemoryStateSplitOperation operation1(2);
  MemoryStateSplitOperation operation2(4);
  TestOperation operation3({ memoryStateType }, { memoryStateType, memoryStateType });

  // Act & Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // Number of results differ
  EXPECT_NE(operation1, operation3); // Operation differs
}

TEST(MemoryStateOperationTests, MemoryStateSplitNormalizeSingleResult)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto & splitNode = MemoryStateSplitOperation::CreateNode(ix, 1);

  auto & ex = jlm::rvsdg::GraphExport::Create(*splitNode.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  jlm::rvsdg::ReduceNode<MemoryStateSplitOperation>(
      MemoryStateSplitOperation::NormalizeSingleResult,
      splitNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 0u);
  EXPECT_EQ(ex.origin(), &ix);
}

TEST(MemoryStateOperationTests, MemoryStateSplitNormalizeNestedSplits)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();
  Graph rvsdg;
  auto & ix = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto & splitNode0 = MemoryStateSplitOperation::CreateNode(ix, 3);
  auto & splitNode1 = MemoryStateSplitOperation::CreateNode(*splitNode0.output(0), 2);
  auto & splitNode2 = MemoryStateSplitOperation::CreateNode(*splitNode0.output(2), 2);

  auto & ex0 = jlm::rvsdg::GraphExport::Create(*splitNode1.output(0), "sn10");
  auto & ex1 = jlm::rvsdg::GraphExport::Create(*splitNode1.output(1), "sn11");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*splitNode0.output(1), "sn01");
  auto & ex3 = jlm::rvsdg::GraphExport::Create(*splitNode2.output(0), "sn20");
  auto & ex4 = jlm::rvsdg::GraphExport::Create(*splitNode2.output(1), "sn21");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  jlm::rvsdg::ReduceNode<MemoryStateSplitOperation>(
      MemoryStateSplitOperation::NormalizeNestedSplits,
      splitNode1);
  jlm::rvsdg::ReduceNode<MemoryStateSplitOperation>(
      MemoryStateSplitOperation::NormalizeNestedSplits,
      splitNode2);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  // We should only have MemoryStateSplit left
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 1u);
  auto [splitNode, splitOperation] =
      TryGetSimpleNodeAndOptionalOp<MemoryStateSplitOperation>(*ex0.origin());
  EXPECT_TRUE(splitNode && splitOperation);

  // We should have 7 outputs:
  // - 2 from splitNode1
  // - 2 from splitNode2
  // - 1 from splitNode0
  // - 1 from splitNode0 -> splitNode1
  // - 1 from splitNode0 -> splitNode2
  EXPECT_EQ(splitNode->noutputs(), 7u);
  EXPECT_EQ(TryGetOwnerNode<SimpleNode>(*ex0.origin()), splitNode);
  EXPECT_EQ(TryGetOwnerNode<SimpleNode>(*ex1.origin()), splitNode);
  EXPECT_EQ(TryGetOwnerNode<SimpleNode>(*ex2.origin()), splitNode);
  EXPECT_EQ(TryGetOwnerNode<SimpleNode>(*ex3.origin()), splitNode);
  EXPECT_EQ(TryGetOwnerNode<SimpleNode>(*ex4.origin()), splitNode);
}

TEST(MemoryStateOperationTests, MemoryStateSplitNormalizeSplitMerge)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();
  Graph rvsdg;
  auto & ix0 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & ix1 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & ix2 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto mergeResult = MemoryStateMergeOperation::Create({ &ix0, &ix1, &ix2 });
  auto & splitNode = MemoryStateSplitOperation::CreateNode(*mergeResult, 3);

  auto & ex0 = jlm::rvsdg::GraphExport::Create(*splitNode.output(0), "ex0");
  auto & ex1 = jlm::rvsdg::GraphExport::Create(*splitNode.output(1), "ex1");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*splitNode.output(2), "ex2");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  jlm::rvsdg::ReduceNode<MemoryStateSplitOperation>(
      MemoryStateSplitOperation::NormalizeSplitMerge,
      splitNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 0u);
  EXPECT_EQ(ex0.origin(), &ix0);
  EXPECT_EQ(ex1.origin(), &ix1);
  EXPECT_EQ(ex2.origin(), &ix2);
}

TEST(MemoryStateOperationTests, MemoryStateMergeEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  MemoryStateMergeOperation operation1(2);
  MemoryStateMergeOperation operation2(4);
  TestOperation operation3({ memoryStateType, memoryStateType }, { memoryStateType });

  // Act & Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // Number of operands differ
  EXPECT_NE(operation1, operation3); // Operation differs
}

TEST(MemoryStateOperationTests, MemoryStateMergeNormalizeSingleOperand)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto & mergeNode = MemoryStateMergeOperation::CreateNode({ &ix });

  auto & ex = jlm::rvsdg::GraphExport::Create(*mergeNode.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateMergeOperation>(
      MemoryStateMergeOperation::NormalizeSingleOperand,
      mergeNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 0u);
  EXPECT_EQ(ex.origin(), &ix);
}

TEST(MemoryStateOperationTests, MemoryStateMergeNormalizeDuplicateOperands)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix0 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x0");
  auto & ix1 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x1");

  auto & node = MemoryStateMergeOperation::CreateNode({ &ix0, &ix0, &ix1, &ix1 });

  auto & ex = jlm::rvsdg::GraphExport::Create(*node.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateMergeOperation>(
      MemoryStateMergeOperation::NormalizeDuplicateOperands,
      node);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 1u);
  auto [mergeNode, mergeOperation] =
      TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(*ex.origin());
  EXPECT_TRUE(mergeNode && mergeOperation);

  EXPECT_EQ(mergeNode->ninputs(), 2u);
}

TEST(MemoryStateOperationTests, MemoryStateMergeNormalizeNestedMerges)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix0 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x0");
  auto & ix1 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x1");
  auto & ix2 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x2");
  auto & ix3 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x3");
  auto & ix4 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x4");

  auto & mergeNode0 = MemoryStateMergeOperation::CreateNode({ &ix0, &ix1 });
  auto & mergeNode1 = MemoryStateMergeOperation::CreateNode({ &ix2, &ix3 });
  auto & mergeNode2 =
      MemoryStateMergeOperation::CreateNode({ mergeNode0.output(0), mergeNode1.output(0), &ix4 });

  auto & ex = jlm::rvsdg::GraphExport::Create(*mergeNode2.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateMergeOperation>(
      MemoryStateMergeOperation::NormalizeNestedMerges,
      mergeNode2);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 1u);
  auto [mergeNode, mergeOperation] =
      TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(*ex.origin());
  EXPECT_TRUE(mergeNode && mergeOperation);

  EXPECT_EQ(mergeNode->ninputs(), 5u);
}

TEST(MemoryStateOperationTests, MemoryStateMergeNormalizeNestedSplits)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix0 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x0");
  auto & ix1 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x1");
  auto & ix2 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x2");

  auto & splitNode0 = MemoryStateSplitOperation::CreateNode(ix0, 2);
  auto & splitNode1 = MemoryStateSplitOperation::CreateNode(ix1, 2);
  auto & mergeNode = MemoryStateMergeOperation::CreateNode({ splitNode0.output(0),
                                                             splitNode0.output(1),
                                                             splitNode1.output(0),
                                                             splitNode1.output(1),
                                                             &ix2 });

  auto & ex = jlm::rvsdg::GraphExport::Create(*mergeNode.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateMergeOperation>(MemoryStateMergeOperation::NormalizeMergeSplit, mergeNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 1u);
  auto [node, mergeOperation] =
      TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(*ex.origin());
  EXPECT_TRUE(node && mergeOperation);

  EXPECT_EQ(node->ninputs(), 5u);
  EXPECT_EQ(node->input(0)->origin(), &ix0);
  EXPECT_EQ(node->input(1)->origin(), &ix0);
  EXPECT_EQ(node->input(2)->origin(), &ix1);
  EXPECT_EQ(node->input(3)->origin(), &ix1);
  EXPECT_EQ(node->input(4)->origin(), &ix2);
}

TEST(MemoryStateOperationTests, MemoryStateJoin_NormalizeSingleOperand)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto & mergeNode = MemoryStateJoinOperation::CreateNode({ &ix });

  auto & ex = GraphExport::Create(*mergeNode.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateJoinOperation>(MemoryStateJoinOperation::NormalizeSingleOperand, mergeNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 0u);
  EXPECT_EQ(ex.origin(), &ix);
}

TEST(MemoryStateOperationTests, MemoryStateJoin_NormalizeDuplicateOperands)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & i0 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "i0");
  auto & i1 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "i1");

  auto & node0 = MemoryStateJoinOperation::CreateNode({ &i0, &i0, &i1, &i1 });
  auto & node1 = MemoryStateJoinOperation::CreateNode({ &i0, &i0, &i0, &i0 });

  auto & x0 = GraphExport::Create(*node0.output(0), "x0");
  auto & x1 = GraphExport::Create(*node1.output(0), "x1");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateJoinOperation>(MemoryStateJoinOperation::NormalizeDuplicateOperands, node0);
  ReduceNode<MemoryStateJoinOperation>(MemoryStateJoinOperation::NormalizeDuplicateOperands, node1);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 1u);

  {
    auto [joinNode, joinOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(*x0.origin());
    EXPECT_TRUE(joinNode && joinOperation);

    EXPECT_EQ(joinNode->ninputs(), 2u);
    EXPECT_EQ(joinNode->input(0)->origin(), &i0);
    EXPECT_EQ(joinNode->input(1)->origin(), &i1);
  }

  {
    EXPECT_EQ(x1.origin(), &i0);
  }
}

TEST(MemoryStateOperationTests, MemoryStateJoin_NormalizeNestedJoins)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix0 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x0");
  auto & ix1 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x1");
  auto & ix2 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x2");
  auto & ix3 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x3");
  auto & ix4 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x4");
  auto & ix5 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x4");

  auto & joinNode0 = MemoryStateJoinOperation::CreateNode({ &ix0, &ix1 });
  auto & joinNode1 = MemoryStateJoinOperation::CreateNode({ joinNode0.output(0), &ix2 });
  auto & joinNode2 = MemoryStateJoinOperation::CreateNode({ &ix3, &ix4 });
  auto & joinNode3 =
      MemoryStateJoinOperation::CreateNode({ joinNode1.output(0), joinNode2.output(0), &ix5 });

  auto & ex = GraphExport::Create(*joinNode3.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateJoinOperation>(MemoryStateJoinOperation::NormalizeNestedJoins, joinNode3);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 1u);
  auto [joinNode, joinOperation] =
      TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(*ex.origin());
  EXPECT_TRUE(joinNode && joinOperation);

  EXPECT_EQ(joinNode->ninputs(), 6u);
  EXPECT_EQ(joinNode->input(0)->origin(), &ix0);
  EXPECT_EQ(joinNode->input(1)->origin(), &ix1);
  EXPECT_EQ(joinNode->input(2)->origin(), &ix2);
  EXPECT_EQ(joinNode->input(3)->origin(), &ix3);
  EXPECT_EQ(joinNode->input(4)->origin(), &ix4);
  EXPECT_EQ(joinNode->input(5)->origin(), &ix5);
}

TEST(MemoryStateOperationTests, LambdaEntryMemStateOperatorEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  const LambdaEntryMemoryStateSplitOperation operation1({ 1, 2 });
  const LambdaEntryMemoryStateSplitOperation operation2({ 3, 4 });
  const LambdaEntryMemoryStateSplitOperation operation3({ 1, 2, 3, 4 });
  const TestOperation operation4({ memoryStateType }, { memoryStateType, memoryStateType });

  // Act & Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // Memory node identifiers differ
  EXPECT_NE(operation1, operation3); // Number of results differ
  EXPECT_NE(operation1, operation4); // Operation differs
}

TEST(MemoryStateOperationTests, LambdaExitMemStateOperatorEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  const LambdaExitMemoryStateMergeOperation operation1({ 1, 2 });
  const LambdaExitMemoryStateMergeOperation operation2({ 3, 4 });
  const LambdaExitMemoryStateMergeOperation operation3({ 1, 2, 3, 4 });
  TestOperation operation4({ memoryStateType, memoryStateType }, { memoryStateType });

  // Act & Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // Memory node identifiers differ
  EXPECT_NE(operation1, operation3); // Number of results differ
  EXPECT_NE(operation1, operation3); // Operation differs
}

TEST(MemoryStateOperationTests, LambdaExitMemoryStateMergeNormalizeLoad)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto bit32Type = BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = TestType::createValueType();

  Graph graph;
  auto & memState1 = jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memState1");
  auto & memState2 = jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memState1");
  auto & size = jlm::rvsdg::GraphImport::Create(graph, bit32Type, "size");

  auto allocaResults = AllocaOperation::create(valueType, &size, 4);
  auto & loadNode =
      LoadNonVolatileOperation::CreateNode(*allocaResults[0], { allocaResults[1] }, valueType, 4);

  auto & lambdaExitMergeNode1 = LambdaExitMemoryStateMergeOperation::CreateNode(
      graph.GetRootRegion(),
      { loadNode.output(1), &memState1 },
      { 1, 2 });

  auto & lambdaExitMergeNode2 = LambdaExitMemoryStateMergeOperation::CreateNode(
      graph.GetRootRegion(),
      { &memState2, &memState1 },
      { 3, 2 });

  auto & x = GraphExport::Create(*lambdaExitMergeNode1.output(0), "x");
  auto & y = GraphExport::Create(*lambdaExitMergeNode2.output(0), "y");
  GraphExport::Create(*loadNode.output(0), "z");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LambdaExitMemoryStateMergeOperation>(
      LambdaExitMemoryStateMergeOperation::NormalizeLoadFromAlloca,
      *jlm::util::assertedCast<SimpleNode>(&lambdaExitMergeNode1));
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_TRUE(success);
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 4u);

  // The lambdaExitMergeNode1 should have been replaced
  const auto [memStateMerge1Node, memStateMerge1Operation] =
      TryGetSimpleNodeAndOptionalOp<LambdaExitMemoryStateMergeOperation>(*x.origin());
  EXPECT_NE(memStateMerge1Node, &lambdaExitMergeNode1);
  EXPECT_EQ(memStateMerge1Node->ninputs(), 2u);
  EXPECT_EQ(memStateMerge1Node->input(0)->origin(), allocaResults[1]);
  EXPECT_EQ(memStateMerge1Node->input(1)->origin(), &memState1);
  EXPECT_EQ(memStateMerge1Operation->getMemoryNodeIds(), std::vector<MemoryNodeId>({ 1, 2 }));

  // The lambdaExitMergeNode2 should not have been replaced
  const auto memStateMerge2Node = TryGetOwnerNode<Node>(*y.origin());
  EXPECT_EQ(memStateMerge2Node, &lambdaExitMergeNode2);
}

TEST(MemoryStateOperationTests, LambdaExitMemoryStateMergeNormalizeStore)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto bit32Type = BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = TestType::createValueType();

  Graph graph;
  auto & memState1 = jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memState1");
  auto & memState2 = jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memState1");
  auto & size = jlm::rvsdg::GraphImport::Create(graph, bit32Type, "size");

  auto allocaResults = AllocaOperation::create(valueType, &size, 4);
  auto & storeNode =
      StoreNonVolatileOperation::CreateNode(*allocaResults[0], size, { allocaResults[1] }, 4);

  auto & lambdaExitMergeNode1 = LambdaExitMemoryStateMergeOperation::CreateNode(
      graph.GetRootRegion(),
      { storeNode.output(0), &memState1 },
      { 1, 2 });

  auto & lambdaExitMergeNode2 = LambdaExitMemoryStateMergeOperation::CreateNode(
      graph.GetRootRegion(),
      { &memState2, &memState1 },
      { 3, 1 });

  auto & x = GraphExport::Create(*lambdaExitMergeNode1.output(0), "x");
  auto & y = GraphExport::Create(*lambdaExitMergeNode2.output(0), "y");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LambdaExitMemoryStateMergeOperation>(
      LambdaExitMemoryStateMergeOperation::NormalizeStoreToAlloca,
      *jlm::util::assertedCast<SimpleNode>(&lambdaExitMergeNode1));
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_TRUE(success);
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 3u);

  // The lambdaExitMergeNode1 should have been replaced
  const auto [memStateMerge1Node, memStateMerge1Operation] =
      TryGetSimpleNodeAndOptionalOp<LambdaExitMemoryStateMergeOperation>(*x.origin());
  EXPECT_NE(memStateMerge1Node, &lambdaExitMergeNode1);
  EXPECT_EQ(memStateMerge1Node->ninputs(), 2u);
  EXPECT_EQ(memStateMerge1Node->input(0)->origin(), allocaResults[1]);
  EXPECT_EQ(memStateMerge1Node->input(1)->origin(), &memState1);
  EXPECT_EQ(memStateMerge1Operation->getMemoryNodeIds(), std::vector<MemoryNodeId>({ 1, 2 }));

  // The lambdaExitMergeNode2 should not have been replaced
  const auto memStateMerge2Node = TryGetOwnerNode<Node>(*y.origin());
  EXPECT_EQ(memStateMerge2Node, &lambdaExitMergeNode2);
}

TEST(MemoryStateOperationTests, LambdaExitMemoryStateMergeNormalizeAlloca)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto bit32Type = BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = TestType::createValueType();

  Graph graph;
  auto & memState1 = jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memState1");
  auto & memState2 = jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memState1");
  auto & size = jlm::rvsdg::GraphImport::Create(graph, bit32Type, "size");

  auto allocaResults = AllocaOperation::create(valueType, &size, 4);

  auto & lambdaExitMergeNode1 = LambdaExitMemoryStateMergeOperation::CreateNode(
      graph.GetRootRegion(),
      { allocaResults[1], &memState1 },
      { 1, 2 });

  auto & lambdaExitMergeNode2 = LambdaExitMemoryStateMergeOperation::CreateNode(
      graph.GetRootRegion(),
      { &memState2, &memState1 },
      { 3, 2 });

  auto & x = GraphExport::Create(*lambdaExitMergeNode1.output(0), "x");
  auto & y = GraphExport::Create(*lambdaExitMergeNode2.output(0), "y");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LambdaExitMemoryStateMergeOperation>(
      LambdaExitMemoryStateMergeOperation::NormalizeAlloca,
      *jlm::util::assertedCast<SimpleNode>(&lambdaExitMergeNode1));
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_TRUE(success);
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 3u);

  // The lambdaExitMergeNode1 should have been replaced
  const auto [memStateMerge1Node, memStateMerge1Operation] =
      TryGetSimpleNodeAndOptionalOp<LambdaExitMemoryStateMergeOperation>(*x.origin());
  EXPECT_NE(memStateMerge1Node, &lambdaExitMergeNode1);
  EXPECT_EQ(memStateMerge1Node->ninputs(), 2u);
  EXPECT_EQ(memStateMerge1Operation->getMemoryNodeIds(), std::vector<MemoryNodeId>({ 1, 2 }));
  const auto undefNode = TryGetOwnerNode<Node>(*memStateMerge1Node->input(0)->origin());
  EXPECT_NE(undefNode, nullptr);
  EXPECT_EQ(memStateMerge1Node->input(1)->origin(), &memState1);

  // The lambdaExitMergeNode2 should not have been replaced
  const auto memStateMerge2Node = TryGetOwnerNode<Node>(*y.origin());
  EXPECT_EQ(memStateMerge2Node, &lambdaExitMergeNode2);
}

TEST(MemoryStateOperationTests, CallEntryMemStateOperatorEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  const CallEntryMemoryStateMergeOperation operation1({ 1, 2 });
  const CallEntryMemoryStateMergeOperation operation2({ 3, 4 });
  const CallEntryMemoryStateMergeOperation operation3({ 1, 2, 3, 4 });
  TestOperation operation4({ memoryStateType, memoryStateType }, { memoryStateType });

  // Act & Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // Memory node identifiers differ
  EXPECT_NE(operation1, operation3); // Number of operands differ
  EXPECT_NE(operation1, operation3); // Operation differs
}

TEST(MemoryStateOperationTests, CallExitMemStateOperatorEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  const CallExitMemoryStateSplitOperation operation1({ 1, 2 });
  const CallExitMemoryStateSplitOperation operation2({ 3, 4 });
  const CallExitMemoryStateSplitOperation operation3({ 1, 2, 3, 4 });
  const TestOperation operation4({ memoryStateType }, { memoryStateType, memoryStateType });

  // Act & Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // Memory node identifiers differ
  EXPECT_NE(operation1, operation3); // Number of memory node identifiers differ
  EXPECT_NE(operation1, operation4); // Operation differs
}
