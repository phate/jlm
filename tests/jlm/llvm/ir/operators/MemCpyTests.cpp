/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/MemCpy.hpp>

static void
OperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto bit32Type = jlm::rvsdg::bittype::Create(32);
  auto bit64Type = jlm::rvsdg::bittype::Create(64);

  MemCpyNonVolatileOperation operation1(bit32Type, 1);
  MemCpyNonVolatileOperation operation2(bit64Type, 4);
  jlm::tests::TestOperation operation3({ valueType }, { valueType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // length type differs
  assert(operation1 != operation3); // number of memory states differs
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemCpyNonVolatileTests-OperationEquality",
    OperationEquality)
