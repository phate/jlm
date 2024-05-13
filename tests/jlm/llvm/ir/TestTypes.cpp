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

  jlm::tests::valuetype valueType;
  PointerType pointerType;
  MemoryStateType memoryStateType;
  iostatetype ioStateType;

  // Direct checks
  assert(IsOrContains<PointerType>(pointerType));
  assert(!IsOrContains<PointerType>(memoryStateType));
  assert(!IsOrContains<PointerType>(ioStateType));

  // Checking supertypes should work
  assert(IsOrContains<jlm::rvsdg::valuetype>(pointerType));
  assert(!IsOrContains<jlm::rvsdg::valuetype>(memoryStateType));
  assert(!IsOrContains<jlm::rvsdg::valuetype>(ioStateType));
  assert(!IsOrContains<jlm::rvsdg::statetype>(pointerType));
  assert(IsOrContains<jlm::rvsdg::statetype>(memoryStateType));
  assert(IsOrContains<jlm::rvsdg::statetype>(ioStateType));

  // Function types are not aggregate types
  FunctionType functionType(
      { &pointerType, &memoryStateType, &ioStateType },
      { &pointerType, &memoryStateType, &ioStateType });
  assert(!IsAggregateType(functionType));
  assert(IsOrContains<FunctionType>(functionType));
  assert(!IsOrContains<PointerType>(functionType));
  assert(!IsOrContains<jlm::rvsdg::statetype>(functionType));

  // Struct types are aggregates that can contain other types
  auto declaration = StructType::Declaration::Create({ &valueType, &pointerType });
  StructType structType(false, *declaration);
  assert(IsAggregateType(structType));
  assert(IsOrContains<StructType>(structType));
  assert(IsOrContains<PointerType>(structType));
  assert(!IsOrContains<jlm::rvsdg::statetype>(structType));

  // Create an array containing the atruct type
  arraytype arrayType(structType, 20);
  assert(IsAggregateType(arrayType));
  assert(IsOrContains<arraytype>(arrayType));
  assert(IsOrContains<StructType>(arrayType));
  assert(IsOrContains<PointerType>(arrayType));
  assert(!IsOrContains<jlm::rvsdg::statetype>(arrayType));

  // Vector types are weird, as LLVM does not consider them to be aggregate types,
  // but they still contain other types
  fixedvectortype vectorType(structType, 20);
  assert(!IsAggregateType(vectorType));
  assert(IsOrContains<vectortype>(vectorType));
  assert(IsOrContains<StructType>(vectorType));
  assert(IsOrContains<PointerType>(vectorType));
  assert(!IsOrContains<jlm::rvsdg::statetype>(vectorType));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TestTypes-TestIsOrContains", TestIsOrContains);
