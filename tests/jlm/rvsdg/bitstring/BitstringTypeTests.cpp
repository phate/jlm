/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/bitstring/type.hpp>

#include <cassert>

static int
TestComputeHash()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  bittype bitType1(8);
  bittype bitType2(16);
  bittype bitType3(8);

  // Act & Assert
  assert(bitType1.ComputeHash() == bitType1.ComputeHash());
  assert(bitType1.ComputeHash() != bitType2.ComputeHash());
  assert(bitType1.ComputeHash() == bitType3.ComputeHash());

  assert(bitType2.ComputeHash() == bitType2.ComputeHash());
  assert(bitType2.ComputeHash() != bitType3.ComputeHash());

  assert(bitType3.ComputeHash() == bitType3.ComputeHash());
  assert(bitType3.ComputeHash() != valueType->ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/BitstringTypeTests-TestComputeHash", TestComputeHash);
