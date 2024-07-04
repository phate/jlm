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

  scalablevectortype scalableVectorType1(valueType, 2);
  scalablevectortype scalableVectorType2(valueType, 3);
  scalablevectortype scalableVectorType3(bitstringType, 2);
  fixedvectortype fixedVectorType(valueType, 2);

  // Act & Assert
  assert(scalableVectorType1.ComputeHash() == scalableVectorType1.ComputeHash());
  assert(scalableVectorType1.ComputeHash() != scalableVectorType2.ComputeHash());
  assert(scalableVectorType1.ComputeHash() != scalableVectorType3.ComputeHash());
  assert(scalableVectorType1.ComputeHash() != fixedVectorType.ComputeHash());

  assert(scalableVectorType2.ComputeHash() == scalableVectorType2.ComputeHash());
  assert(scalableVectorType2.ComputeHash() != scalableVectorType3.ComputeHash());
  assert(scalableVectorType2.ComputeHash() != fixedVectorType.ComputeHash());

  assert(scalableVectorType3.ComputeHash() == scalableVectorType3.ComputeHash());
  assert(scalableVectorType3.ComputeHash() != fixedVectorType.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/ScalableVectorTypeTests-TestComputeHash", TestComputeHash);
