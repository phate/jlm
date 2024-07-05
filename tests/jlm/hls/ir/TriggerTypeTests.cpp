/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/hls/ir/hls.hpp>

#include <cassert>

static int
TestComputeHash()
{
  using namespace jlm::hls;

  // Arrange
  triggertype triggerType1;
  triggertype triggerType2;
  jlm::llvm::iostatetype ioStateType;

  // Act & Assert
  assert(triggerType1.ComputeHash() == triggerType1.ComputeHash());
  assert(triggerType2.ComputeHash() == triggerType2.ComputeHash());
  assert(triggerType1.ComputeHash() != ioStateType.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/ir/TriggerTypeTests-TestComputeHash", TestComputeHash);
