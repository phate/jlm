/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <tests/test-types.hpp>

#include <jlm/llvm/ir/types.hpp>

#include <cassert>

static int
TestComputeHash()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto stateType = jlm::tests::statetype::Create();

  FunctionType functionType1({ valueType }, { stateType });
  FunctionType functionType2({ stateType }, { valueType });
  FunctionType functionType3({ valueType, stateType }, {});
  FunctionType functionType4({ valueType }, { stateType });

  // Act & Assert
  assert(functionType1.ComputeHash() == functionType1.ComputeHash());
  assert(functionType1.ComputeHash() != functionType2.ComputeHash());
  assert(functionType1.ComputeHash() != functionType3.ComputeHash());
  assert(functionType1.ComputeHash() == functionType4.ComputeHash());

  assert(functionType2.ComputeHash() == functionType2.ComputeHash());
  assert(functionType2.ComputeHash() != functionType3.ComputeHash());
  assert(functionType2.ComputeHash() != functionType4.ComputeHash());

  assert(functionType3.ComputeHash() == functionType3.ComputeHash());
  assert(functionType3.ComputeHash() != functionType4.ComputeHash());

  assert(functionType4.ComputeHash() == functionType4.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/FunctionTypeTests-TestComputeHash", TestComputeHash);
