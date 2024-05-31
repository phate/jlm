/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>

static int
MemoryStateSplitEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryStateType;
  MemoryStateSplitOperation operation1(2);
  MemoryStateSplitOperation operation2(4);
  jlm::tests::test_op operation3({ &memoryStateType }, { &memoryStateType, &memoryStateType });

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
MemoryStateMergeEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryStateType;
  MemoryStateMergeOperation operation1(2);
  MemoryStateMergeOperation operation2(4);
  jlm::tests::test_op operation3({ &memoryStateType, &memoryStateType }, { &memoryStateType });

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
  jlm::llvm::MemoryStateType memoryStateType;
  LambdaEntryMemoryStateSplitOperation operation1(2);
  LambdaEntryMemoryStateSplitOperation operation2(4);
  jlm::tests::test_op operation3({ &memoryStateType }, { &memoryStateType, &memoryStateType });

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
  jlm::llvm::MemoryStateType memoryStateType;
  LambdaExitMemStateOperator operation1(2);
  LambdaExitMemStateOperator operation2(4);
  jlm::tests::test_op operation3({ &memoryStateType, &memoryStateType }, { &memoryStateType });

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
  jlm::llvm::MemoryStateType memoryStateType;
  CallEntryMemStateOperator operation1(2);
  CallEntryMemStateOperator operation2(4);
  jlm::tests::test_op operation3({ &memoryStateType, &memoryStateType }, { &memoryStateType });

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
  jlm::llvm::MemoryStateType memoryStateType;
  CallExitMemStateOperator operation1(2);
  CallExitMemStateOperator operation2(4);
  jlm::tests::test_op operation3({ &memoryStateType }, { &memoryStateType, &memoryStateType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // Number of results differ
  assert(operation1 != operation3); // Operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemoryStateOperationTests-CallExitMemStateOperatorEquality",
    CallExitMemStateOperatorEquality)
