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
  MemoryStateType memoryStateType1;
  MemoryStateType memoryStateType2;
  iostatetype ioStateType;

  // Act & Assert
  assert(memoryStateType1.ComputeHash() == memoryStateType1.ComputeHash());
  assert(memoryStateType2.ComputeHash() == memoryStateType2.ComputeHash());
  assert(memoryStateType1.ComputeHash() != ioStateType.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/MemoryStateTypeTests-TestComputeHash", TestComputeHash);
