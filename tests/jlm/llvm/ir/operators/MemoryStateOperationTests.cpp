/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/llvm/ir/RvsdgModule.hpp"
#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

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
NormalizeMemoryStateSplitSingleResultTest()
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
    "jlm/llvm/ir/operators/MemoryStateOperationTests-NormalizeMemoryStateSplitSingleResultTest",
    NormalizeMemoryStateSplitSingleResultTest)

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
