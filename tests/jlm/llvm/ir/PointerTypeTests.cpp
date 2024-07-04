/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/types.hpp>

#include <cassert>

static int
TestComputeHash()
{
  using namespace jlm::llvm;

  // Arrange
  PointerType pointerType1;
  PointerType pointerType2;

  // Act & Assert
  assert(pointerType1.ComputeHash() == pointerType1.ComputeHash());
  assert(pointerType2.ComputeHash() == pointerType2.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/PointerTypeTests-TestComputeHash", TestComputeHash);
