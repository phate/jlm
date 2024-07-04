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
  fptype floatingPointType1(fpsize::half);
  fptype floatingPointType2(fpsize::flt);
  fptype floatingPointType3(fpsize::half);

  // Act & Assert
  assert(floatingPointType1.ComputeHash() == floatingPointType1.ComputeHash());
  assert(floatingPointType1.ComputeHash() != floatingPointType2.ComputeHash());
  assert(floatingPointType1.ComputeHash() == floatingPointType3.ComputeHash());

  assert(floatingPointType2.ComputeHash() == floatingPointType2.ComputeHash());
  assert(floatingPointType2.ComputeHash() == floatingPointType3.ComputeHash());

  assert(floatingPointType3.ComputeHash() == floatingPointType3.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/FloatingPointTypeTests-TestComputeHash", TestComputeHash);
