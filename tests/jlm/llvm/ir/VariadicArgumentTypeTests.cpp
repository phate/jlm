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
  varargtype variadicArgumentType1;
  varargtype variadicArgumentType2;

  // Act & Assert
  assert(variadicArgumentType1.ComputeHash() == variadicArgumentType1.ComputeHash());
  assert(variadicArgumentType1.ComputeHash() == variadicArgumentType2.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/VariadicArgumentTypeTests-TestComputeHash", TestComputeHash);
