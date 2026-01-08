/*
 * Copyright 2024, 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/TestType.hpp>

#include <cassert>

TEST(TypeTests, TestIsOrContains)
{
  using namespace jlm::llvm;

  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto ioStateType = IOStateType::Create();

  // Direct checks
  EXPECT_TRUE(IsOrContains<PointerType>(*pointerType));
  EXPECT_FALSE(IsOrContains<PointerType>(*memoryStateType));
  EXPECT_FALSE(IsOrContains<PointerType>(*ioStateType));

  // Check type kinds
  EXPECT_EQ(pointerType->Kind(), jlm::rvsdg::TypeKind::Value);
  EXPECT_EQ(memoryStateType->Kind(), jlm::rvsdg::TypeKind::State);
  EXPECT_EQ(ioStateType->Kind(), jlm::rvsdg::TypeKind::State);

  // Function types are not aggregate types
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create(), IOStateType::Create() },
      { PointerType::Create(), MemoryStateType::Create(), IOStateType::Create() });
  EXPECT_FALSE(IsAggregateType(*functionType));
  EXPECT_TRUE(IsOrContains<jlm::rvsdg::FunctionType>(*functionType));
  EXPECT_FALSE(IsOrContains<PointerType>(*functionType));
  EXPECT_EQ(functionType->Kind(), jlm::rvsdg::TypeKind::Value);

  // Struct types are aggregates that can contain other types
  auto declaration = StructType::Declaration::Create({ valueType, pointerType });
  auto structType = StructType::Create(false, *declaration);
  EXPECT_TRUE(IsAggregateType(*structType));
  EXPECT_TRUE(IsOrContains<StructType>(*structType));
  EXPECT_TRUE(IsOrContains<PointerType>(*structType));
  EXPECT_EQ(structType->Kind(), jlm::rvsdg::TypeKind::Value);

  // Create an array containing the atruct type
  auto arrayType = ArrayType::Create(structType, 20);
  EXPECT_TRUE(IsAggregateType(*arrayType));
  EXPECT_TRUE(IsOrContains<ArrayType>(*arrayType));
  EXPECT_TRUE(IsOrContains<StructType>(*arrayType));
  EXPECT_TRUE(IsOrContains<PointerType>(*arrayType));
  EXPECT_EQ(arrayType->Kind(), jlm::rvsdg::TypeKind::Value);

  // Vector types are weird, as LLVM does not consider them to be aggregate types,
  // but they still contain other types
  const auto vectorType = FixedVectorType::Create(structType, 20);
  EXPECT_FALSE(IsAggregateType(*vectorType));
  EXPECT_TRUE(IsOrContains<VectorType>(*vectorType));
  EXPECT_TRUE(IsOrContains<StructType>(*vectorType));
  EXPECT_TRUE(IsOrContains<PointerType>(*vectorType));
  EXPECT_EQ(vectorType->Kind(), jlm::rvsdg::TypeKind::Value);
}

TEST(TypeTests, TestGetTypeSizeAndAlignment)
{
  using namespace jlm::llvm;

  auto pointerType = PointerType::Create();
  EXPECT_EQ(GetTypeStoreSize(*pointerType), 8u);
  EXPECT_EQ(GetTypeAllocSize(*pointerType), 8u);
  EXPECT_EQ(GetTypeAlignment(*pointerType), 8u);

  auto bits32 = jlm::rvsdg::BitType::Create(32);
  auto bits50 = jlm::rvsdg::BitType::Create(50);
  EXPECT_EQ(GetTypeStoreSize(*bits32), 4u);
  EXPECT_EQ(GetTypeAllocSize(*bits32), 4u);
  EXPECT_EQ(GetTypeAlignment(*bits32), 4u);

  EXPECT_EQ(GetTypeStoreSize(*bits50), 7u);
  EXPECT_EQ(GetTypeAllocSize(*bits50), 8u);
  EXPECT_EQ(GetTypeAlignment(*bits50), 8u);

  auto floatType = FloatingPointType::Create(fpsize::fp128);
  EXPECT_EQ(GetTypeStoreSize(*floatType), 16u);
  EXPECT_EQ(GetTypeAllocSize(*floatType), 16u);
  EXPECT_EQ(GetTypeAlignment(*floatType), 16u);
  floatType = FloatingPointType::Create(fpsize::half);
  EXPECT_EQ(GetTypeStoreSize(*floatType), 2u);
  EXPECT_EQ(GetTypeAllocSize(*floatType), 2u);
  EXPECT_EQ(GetTypeAlignment(*floatType), 2u);

  auto arrayType = ArrayType::Create(bits32, 5);
  unsigned int expectedArrayStoreSize = 4 * 5;
  unsigned int expectedArrayAllocAize = 4 * 5;
  EXPECT_EQ(GetTypeStoreSize(*arrayType), expectedArrayStoreSize);
  EXPECT_EQ(GetTypeAllocSize(*arrayType), expectedArrayAllocAize);
  EXPECT_EQ(GetTypeAlignment(*arrayType), 4u);

  // Vectors are always aligned up, so a <3 x i32> is 12 bytes of data and 4 bytes of padding.
  // Vectors are also very aligned
  auto vectorType = FixedVectorType::Create(bits32, 3);
  EXPECT_EQ(GetTypeStoreSize(*vectorType), 12u);
  EXPECT_EQ(GetTypeAllocSize(*vectorType), 16u);
  EXPECT_EQ(GetTypeAlignment(*vectorType), 16u);

  auto structDeclaration = StructType::Declaration::Create();
  structDeclaration->Append(bits32);
  structDeclaration->Append(pointerType);
  structDeclaration->Append(arrayType);
  structDeclaration->Append(vectorType); // The most aligned type, 16 byte alignment
  structDeclaration->Append(bits32);

  auto structType = StructType::Create("myStruct", false, *structDeclaration);
  EXPECT_EQ(structType->GetFieldOffset(0), 0u);
  EXPECT_EQ(structType->GetFieldOffset(1), 8u); // Due to 4 bytes of padding after i32
  EXPECT_EQ(structType->GetFieldOffset(2), 16u);
  EXPECT_EQ(structType->GetFieldOffset(3), 48u); // 12 bytes of padding after array
  EXPECT_EQ(structType->GetFieldOffset(4), 64u);
  EXPECT_EQ(GetTypeStoreSize(*structType), 80u); // Struct ends with 12 bytes of padding
  EXPECT_EQ(GetTypeAllocSize(*structType), 80u);
  EXPECT_EQ(GetTypeAlignment(*structType), 16u);

  auto packedStructType = StructType::Create("myPackedStruct", true, *structDeclaration);
  EXPECT_EQ(packedStructType->GetFieldOffset(0), 0u);
  EXPECT_EQ(packedStructType->GetFieldOffset(1), 4u);
  EXPECT_EQ(packedStructType->GetFieldOffset(2), 12u);
  EXPECT_EQ(packedStructType->GetFieldOffset(3), 32u); // array is 20 bytes
  EXPECT_EQ(packedStructType->GetFieldOffset(4), 48u); // vector is 16 bytes
  EXPECT_EQ(GetTypeStoreSize(*packedStructType), 52u);
  EXPECT_EQ(GetTypeAllocSize(*packedStructType), 52u);
  EXPECT_EQ(GetTypeAlignment(*packedStructType), 1u);
}
