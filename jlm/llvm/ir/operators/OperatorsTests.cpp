/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>

namespace jlm::llvm
{

TEST(PtrCmpOperationTests, testNormalizeNullPointerComparison)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto pointerType = PointerType::Create();
  auto i32Type = BitType::Create(32);
  auto functionType = FunctionType::Create({}, { pointerType });

  Graph graph;

  auto & i0 = LlvmGraphImport::create(
      graph,
      i32Type,
      pointerType,
      "i0",
      Linkage::externalLinkage,
      CallingConvention::C,
      true,
      4);

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      LlvmDeltaOperation::Create(pointerType, "delta", Linkage::externalLinkage, "", true, 4));
  auto & ptrNullDeltaNode = ConstantPointerNullOperation::createNode(*deltaNode->subregion());
  auto & deltaOutput = deltaNode->finalize(ptrNullDeltaNode.output(0));

  auto lambdaNode = LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          functionType,
          "lambda",
          Linkage::externalLinkage,
          CallingConvention::C,
          {}));
  auto & ptrNullLambdaNode = ConstantPointerNullOperation::createNode(*lambdaNode->subregion());
  auto lambdaOutput = lambdaNode->finalize({ ptrNullLambdaNode.output(0) });

  auto & fnToPtrNode = CreateOpNode<FunctionToPointerOperation>({ lambdaOutput }, functionType);

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

  auto & ptrCmpNode3 = PtrCmpOperation::createNode(ICmpPredicate::Ne, i0, *cPtrNullNode.output(0));

  auto & ptrCmpNode4 =
      PtrCmpOperation::createNode(ICmpPredicate::Ne, deltaOutput, *cPtrNullNode.output(0));

  auto & ptrCmpNode5 = PtrCmpOperation::createNode(
      ICmpPredicate::Ne,
      *fnToPtrNode.output(0),
      *cPtrNullNode.output(0));

  auto & x1 = GraphExport::Create(*ptrCmpNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*ptrCmpNode2.output(0), "x2");
  auto & x3 = GraphExport::Create(*ptrCmpNode3.output(0), "x3");
  auto & x4 = GraphExport::Create(*ptrCmpNode4.output(0), "x4");
  auto & x5 = GraphExport::Create(*ptrCmpNode5.output(0), "x5");

  // Act
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode1);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode2);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode3);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode4);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode5);

  // Assert
  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x1.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1u);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 0u);
  }

  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x2.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1u);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 1u);
  }

  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x3.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1u);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 1u);
  }

  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x4.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1u);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 1u);
  }

  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x5.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1u);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 1u);
  }
}

}
