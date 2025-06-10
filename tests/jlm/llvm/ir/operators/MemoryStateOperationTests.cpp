/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>
#include <llvm-18/llvm/ADT/StringExtras.h>

static int
MemoryStateSplitEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  MemoryStateSplitOperation operation1(2);
  MemoryStateSplitOperation operation2(4);
  jlm::tests::test_op operation3({ memoryStateType }, { memoryStateType, memoryStateType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // Number of results differ
  assert(operation1 != operation3); // Operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateSplitEquality",
    MemoryStateSplitEquality)

static int
MemoryStateSplitNormalizeSingleResult()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto & splitNode = MemoryStateSplitOperation::CreateNode(ix, 1);

  auto & ex = jlm::tests::GraphExport::Create(*splitNode.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  jlm::rvsdg::ReduceNode<MemoryStateSplitOperation>(
      MemoryStateSplitOperation::NormalizeSingleResult,
      splitNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 0);
  assert(ex.origin() == &ix);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateSplitNormalizeSingleResult",
    MemoryStateSplitNormalizeSingleResult)

static int
MemoryStateSplitNormalizeNestedSplits()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();
  Graph rvsdg;
  auto & ix = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto & splitNode0 = MemoryStateSplitOperation::CreateNode(ix, 3);
  auto & splitNode1 = MemoryStateSplitOperation::CreateNode(*splitNode0.output(0), 2);
  auto & splitNode2 = MemoryStateSplitOperation::CreateNode(*splitNode0.output(2), 2);

  auto & ex0 = jlm::tests::GraphExport::Create(*splitNode1.output(0), "sn10");
  auto & ex1 = jlm::tests::GraphExport::Create(*splitNode1.output(1), "sn11");
  auto & ex2 = jlm::tests::GraphExport::Create(*splitNode0.output(1), "sn01");
  auto & ex3 = jlm::tests::GraphExport::Create(*splitNode2.output(0), "sn20");
  auto & ex4 = jlm::tests::GraphExport::Create(*splitNode2.output(1), "sn21");

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
  assert(rvsdg.GetRootRegion().nnodes() == 1);
  auto [splitNode, splitOperation] =
      TryGetSimpleNodeAndOp<MemoryStateSplitOperation>(*ex0.origin());
  assert(splitNode && splitOperation);

  // We should have 7 outputs:
  // - 2 from splitNode1
  // - 2 from splitNode2
  // - 1 from splitNode0
  // - 1 from splitNode0 -> splitNode1
  // - 1 from splitNode0 -> splitNode2
  assert(splitNode->noutputs() == 7);
  assert(TryGetOwnerNode<SimpleNode>(*ex0.origin()) == splitNode);
  assert(TryGetOwnerNode<SimpleNode>(*ex1.origin()) == splitNode);
  assert(TryGetOwnerNode<SimpleNode>(*ex2.origin()) == splitNode);
  assert(TryGetOwnerNode<SimpleNode>(*ex3.origin()) == splitNode);
  assert(TryGetOwnerNode<SimpleNode>(*ex4.origin()) == splitNode);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateSplitNormalizeNestedSplits",
    MemoryStateSplitNormalizeNestedSplits)

static int
MemoryStateSplitNormalizeSplitMerge()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();
  Graph rvsdg;
  auto & ix0 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & ix1 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & ix2 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto mergeResult = MemoryStateMergeOperation::Create({ &ix0, &ix1, &ix2 });
  auto & splitNode = MemoryStateSplitOperation::CreateNode(*mergeResult, 3);

  auto & ex0 = jlm::tests::GraphExport::Create(*splitNode.output(0), "ex0");
  auto & ex1 = jlm::tests::GraphExport::Create(*splitNode.output(1), "ex1");
  auto & ex2 = jlm::tests::GraphExport::Create(*splitNode.output(2), "ex2");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  jlm::rvsdg::ReduceNode<MemoryStateSplitOperation>(
      MemoryStateSplitOperation::NormalizeSplitMerge,
      splitNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 0);
  assert(ex0.origin() == &ix0);
  assert(ex1.origin() == &ix1);
  assert(ex2.origin() == &ix2);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateSplitNormalizeSplitMerge",
    MemoryStateSplitNormalizeSplitMerge)

static int
MemoryStateMergeEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  MemoryStateMergeOperation operation1(2);
  MemoryStateMergeOperation operation2(4);
  jlm::tests::test_op operation3({ memoryStateType, memoryStateType }, { memoryStateType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // Number of operands differ
  assert(operation1 != operation3); // Operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateMergeEquality",
    MemoryStateMergeEquality)

static int
MemoryStateMergeNormalizeSingleOperand()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x");

  auto & mergeNode = MemoryStateMergeOperation::CreateNode({ &ix });

  auto & ex = jlm::tests::GraphExport::Create(*mergeNode.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateMergeOperation>(
      MemoryStateMergeOperation::NormalizeSingleOperand,
      mergeNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 0);
  assert(ex.origin() == &ix);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateMergeNormalizeSingleOperand",
    MemoryStateMergeNormalizeSingleOperand)

static int
MemoryStateMergeNormalizeDuplicateOperands()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix0 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x0");
  auto & ix1 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x1");

  auto & node = MemoryStateMergeOperation::CreateNode({ &ix0, &ix0, &ix1, &ix1 });

  auto & ex = jlm::tests::GraphExport::Create(*node.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateMergeOperation>(
      MemoryStateMergeOperation::NormalizeDuplicateOperands,
      node);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 1);
  auto [mergeNode, mergeOperation] = TryGetSimpleNodeAndOp<MemoryStateMergeOperation>(*ex.origin());
  assert(mergeNode && mergeOperation);

  assert(mergeNode->ninputs() == 2);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateMergeNormalizeDuplicateOperands",
    MemoryStateMergeNormalizeDuplicateOperands)

static int
MemoryStateMergeNormalizeNestedMerges()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix0 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x0");
  auto & ix1 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x1");
  auto & ix2 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x2");
  auto & ix3 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x3");
  auto & ix4 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x4");

  auto & mergeNode0 = MemoryStateMergeOperation::CreateNode({ &ix0, &ix1 });
  auto & mergeNode1 = MemoryStateMergeOperation::CreateNode({ &ix2, &ix3 });
  auto & mergeNode2 =
      MemoryStateMergeOperation::CreateNode({ mergeNode0.output(0), mergeNode1.output(0), &ix4 });

  auto & ex = jlm::tests::GraphExport::Create(*mergeNode2.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateMergeOperation>(
      MemoryStateMergeOperation::NormalizeNestedMerges,
      mergeNode2);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 1);
  auto [mergeNode, mergeOperation] = TryGetSimpleNodeAndOp<MemoryStateMergeOperation>(*ex.origin());
  assert(mergeNode && mergeOperation);

  assert(mergeNode->ninputs() == 5);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateMergeNormalizeNestedMerges",
    MemoryStateMergeNormalizeNestedMerges)

static int
MemoryStateMergeNormalizeNestedSplits()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();

  Graph rvsdg;
  auto & ix0 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x0");
  auto & ix1 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x1");
  auto & ix2 = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x2");

  auto & splitNode0 = MemoryStateSplitOperation::CreateNode(ix0, 2);
  auto & splitNode1 = MemoryStateSplitOperation::CreateNode(ix1, 2);
  auto & mergeNode = MemoryStateMergeOperation::CreateNode({ splitNode0.output(0),
                                                             splitNode0.output(1),
                                                             splitNode1.output(0),
                                                             splitNode1.output(1),
                                                             &ix2 });

  auto & ex = jlm::tests::GraphExport::Create(*mergeNode.output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  ReduceNode<MemoryStateMergeOperation>(MemoryStateMergeOperation::NormalizeMergeSplit, mergeNode);
  rvsdg.PruneNodes();
  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 1);
  auto [node, mergeOperation] = TryGetSimpleNodeAndOp<MemoryStateMergeOperation>(*ex.origin());
  assert(node && mergeOperation);

  assert(node->ninputs() == 5);
  assert(node->input(0)->origin() == &ix0);
  assert(node->input(1)->origin() == &ix0);
  assert(node->input(2)->origin() == &ix1);
  assert(node->input(3)->origin() == &ix1);
  assert(node->input(4)->origin() == &ix2);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-MemoryStateMergeNormalizeNestedSplits",
    MemoryStateMergeNormalizeNestedSplits)

static int
LambdaEntryMemStateOperatorEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  LambdaEntryMemoryStateSplitOperation operation1(2);
  LambdaEntryMemoryStateSplitOperation operation2(4);
  jlm::tests::test_op operation3({ memoryStateType }, { memoryStateType, memoryStateType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // Number of results differ
  assert(operation1 != operation3); // Operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/MemoryStateOperationTests-LambdaEntryMemStateOperatorEquality",
    LambdaEntryMemStateOperatorEquality)

static int
LambdaExitMemStateOperatorEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  LambdaExitMemoryStateMergeOperation operation1(2);
  LambdaExitMemoryStateMergeOperation operation2(4);
  jlm::tests::test_op operation3({ memoryStateType, memoryStateType }, { memoryStateType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // Number of operands differ
  assert(operation1 != operation3); // Operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-LambdaExitMemStateOperatorEquality",
    LambdaExitMemStateOperatorEquality)

static int
CallEntryMemStateOperatorEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  CallEntryMemoryStateMergeOperation operation1(2);
  CallEntryMemoryStateMergeOperation operation2(4);
  jlm::tests::test_op operation3({ memoryStateType, memoryStateType }, { memoryStateType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // Number of operands differ
  assert(operation1 != operation3); // Operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/MemoryStateOperationTests-CallEntryMemStateOperatorEquality",
    CallEntryMemStateOperatorEquality)

static int
CallExitMemStateOperatorEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  CallExitMemoryStateSplitOperation operation1(2);
  CallExitMemoryStateSplitOperation operation2(4);
  jlm::tests::test_op operation3({ memoryStateType }, { memoryStateType, memoryStateType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // Number of results differ
  assert(operation1 != operation3); // Operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-CallExitMemStateOperatorEquality",
    CallExitMemStateOperatorEquality)
