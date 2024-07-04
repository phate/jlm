/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <tests/test-types.hpp>

#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>

#include <cassert>

static int
TestComputeHash()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto bitstringType = jlm::rvsdg::bittype::Create(8);

  arraytype arrayType1(valueType, 2);
  arraytype arrayType2(valueType, 3);
  arraytype arrayType3(valueType, 2);
  arraytype arrayType4(bitstringType, 2);

  // Act & Assert
  assert(arrayType1.ComputeHash() == arrayType1.ComputeHash());
  assert(arrayType1.ComputeHash() != arrayType2.ComputeHash());
  assert(arrayType1.ComputeHash() == arrayType3.ComputeHash());
  assert(arrayType1.ComputeHash() != arrayType4.ComputeHash());

  assert(arrayType2.ComputeHash() == arrayType2.ComputeHash());
  assert(arrayType2.ComputeHash() != arrayType3.ComputeHash());
  assert(arrayType2.ComputeHash() != arrayType4.ComputeHash());

  assert(arrayType3.ComputeHash() == arrayType3.ComputeHash());
  assert(arrayType3.ComputeHash() != arrayType4.ComputeHash());

  assert(arrayType4.ComputeHash() == arrayType4.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/ArrayTypeTests-TestComputeHash", TestComputeHash);
