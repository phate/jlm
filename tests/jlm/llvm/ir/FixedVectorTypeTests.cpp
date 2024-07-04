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

  fixedvectortype fixedVectorType1(valueType, 2);
  fixedvectortype fixedVectorType2(valueType, 3);
  fixedvectortype fixedVectorType3(bitstringType, 2);
  scalablevectortype scalableVectorType(valueType, 2);

  // Act & Assert
  assert(fixedVectorType1.ComputeHash() == fixedVectorType1.ComputeHash());
  assert(fixedVectorType1.ComputeHash() != fixedVectorType2.ComputeHash());
  assert(fixedVectorType1.ComputeHash() != fixedVectorType3.ComputeHash());
  assert(fixedVectorType1.ComputeHash() != scalableVectorType.ComputeHash());

  assert(fixedVectorType2.ComputeHash() == fixedVectorType2.ComputeHash());
  assert(fixedVectorType2.ComputeHash() != fixedVectorType3.ComputeHash());
  assert(fixedVectorType2.ComputeHash() != scalableVectorType.ComputeHash());

  assert(fixedVectorType3.ComputeHash() == fixedVectorType3.ComputeHash());
  assert(fixedVectorType3.ComputeHash() != scalableVectorType.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/FixedVectorTypeTests-TestComputeHash", TestComputeHash);
