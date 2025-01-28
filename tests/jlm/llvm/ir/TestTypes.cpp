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
TestIsOrContains()
{
  using namespace jlm::llvm;

  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto ioStateType = iostatetype::Create();

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
      { PointerType::Create(), MemoryStateType::Create(), iostatetype::Create() },
      { PointerType::Create(), MemoryStateType::Create(), iostatetype::Create() });
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
  auto arrayType = arraytype::Create(structType, 20);
  assert(IsAggregateType(*arrayType));
  assert(IsOrContains<arraytype>(*arrayType));
  assert(IsOrContains<StructType>(*arrayType));
  assert(IsOrContains<PointerType>(*arrayType));
  assert(!IsOrContains<jlm::rvsdg::StateType>(*arrayType));

  // Vector types are weird, as LLVM does not consider them to be aggregate types,
  // but they still contain other types
  const auto vectorType = FixedVectorType::Create(structType, 20);
  assert(!IsAggregateType(*vectorType));
  assert(IsOrContains<vectortype>(*vectorType));
  assert(IsOrContains<StructType>(*vectorType));
  assert(IsOrContains<PointerType>(*vectorType));
  assert(!IsOrContains<jlm::rvsdg::StateType>(*vectorType));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TestTypes-TestIsOrContains", TestIsOrContains);
