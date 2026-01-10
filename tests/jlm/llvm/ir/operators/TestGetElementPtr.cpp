/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>

TEST(GetElementPtrOperationTests, TestOperationEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto arrayType = ArrayType::Create(BitType::Create(8), 11);

  auto structType1 = StructType::Create(false, { BitType::Create(64), BitType::Create(64) });
  auto structType2 = StructType::Create("myStructType", false, { arrayType, BitType::Create(32) });

  GetElementPtrOperation operation1(
      { jlm::rvsdg::BitType::Create(32), jlm::rvsdg::BitType::Create(32) },
      structType1);
  GetElementPtrOperation operation2(
      { jlm::rvsdg::BitType::Create(32), jlm::rvsdg::BitType::Create(32) },
      structType2);

  EXPECT_NE(operation1, operation2);
}
