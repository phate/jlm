/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>

TEST(GetElementPtrOperationTests, TestOperationEquality)
{
  using namespace jlm::llvm;

  auto arrayType = ArrayType::Create(jlm::rvsdg::BitType::Create(8), 11);

  auto declaration1 = StructType::Declaration::Create(
      { jlm::rvsdg::BitType::Create(64), jlm::rvsdg::BitType::Create(64) });
  auto declaration2 =
      StructType::Declaration::Create({ arrayType, jlm::rvsdg::BitType::Create(32) });

  auto structType1 = StructType::Create(false, *declaration1);
  auto structType2 = StructType::Create("myStructType", false, *declaration2);

  GetElementPtrOperation operation1(
      { jlm::rvsdg::BitType::Create(32), jlm::rvsdg::BitType::Create(32) },
      structType1);
  GetElementPtrOperation operation2(
      { jlm::rvsdg::BitType::Create(32), jlm::rvsdg::BitType::Create(32) },
      structType2);

  EXPECT_NE(operation1, operation2);
}
