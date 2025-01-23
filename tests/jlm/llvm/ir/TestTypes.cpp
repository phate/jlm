/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>
#include <tests/test-types.hpp>

#include <jlm/llvm/ir/types.hpp>

#include <cassert>

static int
IntegerTypeTests()
{
  using namespace jlm::llvm;

  // Arrange & Act
  const auto i1 = IntegerType::Create(1);
  const auto i8 = IntegerType::Create(8);
  const auto i32 = IntegerType::Create(32);

  // Assert
  assert(i1->NumBits() == 1);
  assert(i8->NumBits() == 8);
  assert(i32->NumBits() == 32);

  assert(*i1 != *i8);
  assert(*i8 != *i32);
  const auto i1Tmp = IntegerType::Create(1);
  assert(*i1 == *i1Tmp);

  assert(i1->ComputeHash() == i1Tmp->ComputeHash());
  assert(i1->ComputeHash() != i8->ComputeHash());
  assert(i8->ComputeHash() != i32->ComputeHash());

  assert(i1->debug_string() == "i1");
  assert(i8->debug_string() == "i8");
  assert(i32->debug_string() == "i32");

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TestTypes-IntegerTypeTests", IntegerTypeTests);

static int
TestIsOrContains()
{
  using namespace jlm::llvm;

  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto ioStateType = IOStateType::Create();

  // Direct checks
  assert(IsOrContains<PointerType>(*pointerType));
  assert(!IsOrContains<PointerType>(*memoryStateType));
  assert(!IsOrContains<PointerType>(*ioStateType));

  // Checking supertypes should work
  assert(IsOrContains<jlm::rvsdg::ValueType>(*pointerType));
  assert(!IsOrContains<jlm::rvsdg::ValueType>(*memoryStateType));
  assert(!IsOrContains<jlm::rvsdg::ValueType>(*ioStateType));
  assert(!IsOrContains<jlm::rvsdg::StateType>(*pointerType));
  assert(IsOrContains<jlm::rvsdg::StateType>(*memoryStateType));
  assert(IsOrContains<jlm::rvsdg::StateType>(*ioStateType));

  // Function types are not aggregate types
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create(), IOStateType::Create() },
      { PointerType::Create(), MemoryStateType::Create(), IOStateType::Create() });
  assert(!IsAggregateType(*functionType));
  assert(IsOrContains<jlm::rvsdg::FunctionType>(*functionType));
  assert(!IsOrContains<PointerType>(*functionType));
  assert(!IsOrContains<jlm::rvsdg::StateType>(*functionType));

  // Struct types are aggregates that can contain other types
  auto declaration = StructType::Declaration::Create({ valueType, pointerType });
  auto structType = StructType::Create(false, *declaration);
  assert(IsAggregateType(*structType));
  assert(IsOrContains<StructType>(*structType));
  assert(IsOrContains<PointerType>(*structType));
  assert(!IsOrContains<jlm::rvsdg::StateType>(*structType));

  // Create an array containing the atruct type
  auto arrayType = ArrayType::Create(structType, 20);
  assert(IsAggregateType(*arrayType));
  assert(IsOrContains<ArrayType>(*arrayType));
  assert(IsOrContains<StructType>(*arrayType));
  assert(IsOrContains<PointerType>(*arrayType));
  assert(!IsOrContains<jlm::rvsdg::StateType>(*arrayType));

  // Vector types are weird, as LLVM does not consider them to be aggregate types,
  // but they still contain other types
  const auto vectorType = FixedVectorType::Create(structType, 20);
  assert(!IsAggregateType(*vectorType));
  assert(IsOrContains<VectorType>(*vectorType));
  assert(IsOrContains<StructType>(*vectorType));
  assert(IsOrContains<PointerType>(*vectorType));
  assert(!IsOrContains<jlm::rvsdg::StateType>(*vectorType));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TestTypes-TestIsOrContains", TestIsOrContains);
