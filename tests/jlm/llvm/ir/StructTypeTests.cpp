/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <tests/test-types.hpp>

#include <jlm/llvm/ir/types.hpp>

#include <cassert>

static int
TestComputeHash()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto stateType = jlm::tests::statetype::Create();

  auto declaration1 = StructType::Declaration::Create({ valueType });
  auto declaration2 = StructType::Declaration::Create({ stateType });
  auto declaration3 = StructType::Declaration::Create({ valueType, valueType });

  StructType structType1("structType1", false, *declaration1);
  StructType structType2(false, *declaration1);
  StructType structType3("structType3", true, *declaration2);
  StructType structType4(false, *declaration3);
  StructType structType5(false, *declaration3);

  // Act & Assert
  assert(structType1.ComputeHash() == structType1.ComputeHash());
  assert(structType1.ComputeHash() != structType2.ComputeHash());
  assert(structType1.ComputeHash() != structType3.ComputeHash());
  assert(structType1.ComputeHash() != structType4.ComputeHash());
  assert(structType1.ComputeHash() != structType5.ComputeHash());

  assert(structType2.ComputeHash() == structType2.ComputeHash());
  assert(structType2.ComputeHash() != structType3.ComputeHash());
  assert(structType2.ComputeHash() != structType4.ComputeHash());
  assert(structType2.ComputeHash() != structType5.ComputeHash());

  assert(structType3.ComputeHash() == structType3.ComputeHash());
  assert(structType3.ComputeHash() != structType4.ComputeHash());
  assert(structType3.ComputeHash() != structType5.ComputeHash());

  assert(structType4.ComputeHash() == structType4.ComputeHash());
  assert(structType4.ComputeHash() == structType5.ComputeHash());

  assert(structType5.ComputeHash() == structType5.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/StructTypeTests-TestComputeHash", TestComputeHash);
