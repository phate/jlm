/*
 * Copyright 2024, 2025 Håvard Krogstie <krogstie.havard@gmail.com>
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
JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TestTypes-TestIsOrContains", TestIsOrContains)

static int
TestGetTypeSizeAndAlignment()
{
  using namespace jlm::llvm;

  auto pointerType = PointerType::Create();
  assert(GetTypeSize(*pointerType) == 8);
  assert(GetTypeAlignment(*pointerType) == 8);

  auto bits32 = jlm::rvsdg::bittype::Create(32);
  auto bits50 = jlm::rvsdg::bittype::Create(50);
  assert(GetTypeSize(*bits32) == 4);
  assert(GetTypeAlignment(*bits32) == 4);

  assert(GetTypeSize(*bits50) == 8);
  assert(GetTypeAlignment(*bits50) == 8);

  auto floatType = FloatingPointType::Create(fpsize::fp128);
  assert(GetTypeSize(*floatType) == 16);
  assert(GetTypeAlignment(*floatType) == 16);
  floatType = FloatingPointType::Create(fpsize::half);
  assert(GetTypeSize(*floatType) == 2);
  assert(GetTypeAlignment(*floatType) == 2);

  auto arrayType = ArrayType::Create(bits32, 5);
  assert(GetTypeSize(*arrayType) == 4 * 5);
  assert(GetTypeAlignment(*arrayType) == 4);

  // Vectors are always aligned up, so a <3 x i32> is 12 bytes of data and 4 bytes of padding.
  // Vectors are also very aligned
  auto vectorType = FixedVectorType::Create(bits32, 3);
  assert(GetTypeSize(*vectorType) == 16);
  assert(GetTypeAlignment(*vectorType) == 16);

  auto structDeclaration = StructType::Declaration::Create();
  structDeclaration->Append(bits32);
  structDeclaration->Append(pointerType);
  structDeclaration->Append(arrayType);
  structDeclaration->Append(vectorType); // The most aligned type, 16 byte alignment
  structDeclaration->Append(bits32);

  auto structType = StructType::Create("myStruct", false, *structDeclaration);
  assert(GetStructFieldOffset(*structType, 0) == 0);
  assert(GetStructFieldOffset(*structType, 1) == 8); // Due to 4 bytes of padding after i32
  assert(GetStructFieldOffset(*structType, 2) == 16);
  assert(GetStructFieldOffset(*structType, 3) == 48); // 12 bytes of padding after array
  assert(GetStructFieldOffset(*structType, 3) == 64);
  assert(GetTypeSize(*structType) == 80); // Struct ends with 12 bytes of padding
  assert(GetTypeAlignment(*structType) == 16);

  auto packedStructType = StructType::Create("myPackedStruct", true, *structDeclaration);
  assert(GetStructFieldOffset(*packedStructType, 0) == 0);
  assert(GetStructFieldOffset(*packedStructType, 1) == 4);
  assert(GetStructFieldOffset(*packedStructType, 2) == 12);
  assert(GetStructFieldOffset(*packedStructType, 3) == 32); // array is 20 bytes
  assert(GetStructFieldOffset(*packedStructType, 3) == 48); // vector is 16 bytes
  assert(GetTypeSize(*packedStructType) == 52);
  assert(GetTypeAlignment(*packedStructType) == 1);

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TestTypes-TestGetTypeSizeAndAlignment",
    TestGetTypeSizeAndAlignment)
