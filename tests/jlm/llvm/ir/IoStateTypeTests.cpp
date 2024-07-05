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
  iostatetype ioStateType1;
  iostatetype ioStateType2;
  MemoryStateType memoryStateType;

  // Act & Assert
  assert(ioStateType1.ComputeHash() == ioStateType1.ComputeHash());
  assert(ioStateType2.ComputeHash() == ioStateType2.ComputeHash());
  assert(ioStateType1.ComputeHash() != memoryStateType.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/IoStateTypeTests-TestComputeHash", TestComputeHash);
