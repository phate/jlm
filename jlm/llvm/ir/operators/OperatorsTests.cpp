/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>

namespace jlm::llvm
{

TEST(PtrCmpOperationTests, testNormalizeNullPointerComparison)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i32Type = BitType::Create(32);

  Graph graph;

  auto & oneNode = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1);
  auto & allocaNode = AllocaOperation::createNode(i32Type, *oneNode.output(0), 4);

  auto & cPtrNullNode = ConstantPointerNullOperation::createNode(graph.GetRootRegion());

  auto & ptrCmpNode1 = PtrCmpOperation::createNode(
      ICmpPredicate::Eq,
      AllocaOperation::getPointerOutput(allocaNode),
      *cPtrNullNode.output(0));

  auto & ptrCmpNode2 = PtrCmpOperation::createNode(
      ICmpPredicate::Ne,
      AllocaOperation::getPointerOutput(allocaNode),
      *cPtrNullNode.output(0));

  auto & x1 = GraphExport::Create(*ptrCmpNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*ptrCmpNode2.output(0), "x2");

  // Act
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode1);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode2);

  // Assert
  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x1.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 0);
  }

  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x2.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 1);
  }
}

}
