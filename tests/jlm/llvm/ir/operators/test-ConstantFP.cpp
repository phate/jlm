/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/operators.hpp>

TEST(ConstantFPOperationTests, test_equality)
{
  using namespace jlm::llvm;

  ConstantFP c1(fpsize::half, llvm::APFloat(0.0));
  ConstantFP c2(fpsize::flt, llvm::APFloat(0.0));
  ConstantFP c3(fpsize::flt, llvm::APFloat(-0.0));

  EXPECT_NE(c1, c2);
  EXPECT_NE(c2, c3);
}
