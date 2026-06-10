/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/rvsdg/graph.hpp>

TEST(GetElementPtrOperationTests, TestOperationEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto arrayType = ArrayType::Create(BitType::Create(8), 11);

  auto structType1 = StructType::CreateLiteral({ BitType::Create(64), BitType::Create(64) }, false);
  auto structType2 =
      StructType::CreateIdentified("myStructType", { arrayType, BitType::Create(32) }, false);

  GetElementPtrOperation operation1(
      { jlm::rvsdg::BitType::Create(32), jlm::rvsdg::BitType::Create(32) },
      structType1);
  GetElementPtrOperation operation2(
      { jlm::rvsdg::BitType::Create(32), jlm::rvsdg::BitType::Create(32) },
      structType2);

  EXPECT_NE(operation1, operation2);
}

TEST(GetElementPtrTests, GetTypeOffsetTest)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto bits8Type = BitType::Create(32);
  const auto bits16Type = BitType::Create(32);
  const auto bits32Type = BitType::Create(32);
  const auto pointerType = PointerType::Create();

  auto structType =
      StructType::CreateIdentified("struct", { bits8Type, bits16Type, bits32Type }, false);

  Graph rvsdg;
  auto & baseAddress = GraphImport::Create(rvsdg, pointerType, "base");
  auto & i32 = GraphImport::Create(rvsdg, bits8Type, "i32");

  auto & zeroNode = IntegerConstantOperation::Create(rvsdg.GetRootRegion(), 32, 0);
  auto & oneNode = IntegerConstantOperation::Create(rvsdg.GetRootRegion(), 32, 1);

  auto & gepNode0 = GetElementPtrOperation::createNode(
      baseAddress,
      { zeroNode.output(0), oneNode.output(0) },
      structType);
  auto & gepNode1 =
      GetElementPtrOperation::createNode(baseAddress, { zeroNode.output(0), &i32 }, structType);
  auto & gepNode2 = GetElementPtrOperation::createNode(baseAddress, { &i32, &i32 }, structType);

  // Act
  auto typeOffset0 = GetElementPtrOperation::getTypeOffset(gepNode0);
  auto typeOffset1 = GetElementPtrOperation::getTypeOffset(gepNode1);
  auto typeOffset2 = GetElementPtrOperation::getTypeOffset(gepNode2);

  // Assert
  // We expect typeOffset0 to be present as the \ref GetElementPtrOperation is statically completely
  // known. All others should result in std::nullopt.
  EXPECT_TRUE(typeOffset0.has_value());
  EXPECT_EQ(typeOffset0.value().pointeeType, structType);
  EXPECT_EQ(typeOffset0.value().indices, std::vector<uint64_t>({ 0, 1 }));

  EXPECT_FALSE(typeOffset1.has_value());
  EXPECT_FALSE(typeOffset2.has_value());
}
