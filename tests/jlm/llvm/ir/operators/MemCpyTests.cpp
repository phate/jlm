/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/MemCpy.hpp>

static int
OperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::tests::valuetype valueType;
  jlm::rvsdg::bittype bit32Type(32);
  jlm::rvsdg::bittype bit64Type(64);

  MemCpyNonVolatileOperation operation1(bit32Type, 1);
  MemCpyNonVolatileOperation operation2(bit64Type, 4);
  jlm::tests::test_op operation3({ &valueType }, { &valueType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // length type differs
  assert(operation1 != operation3); // number of memory states differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemCpyNonVolatileTests-OperationEquality",
    OperationEquality)
