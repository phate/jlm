/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>

static int
MemoryStateSplitEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryStateType;
  MemStateSplitOperator operation1(2);
  MemStateSplitOperator operation2(4);
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
  MemStateMergeOperator operation1(2);
  MemStateMergeOperator operation2(4);
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
