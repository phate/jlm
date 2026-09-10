/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <llvm/ADT/APFloat.h>

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

  auto & ptrCmpNode6 = PtrCmpOperation::createNode(
      ICmpPredicate::Eq,
      *cPtrNullNode.output(0),
      *cPtrNullNode.output(0));

  auto & ptrCmpNode7 = PtrCmpOperation::createNode(
      ICmpPredicate::Ne,
      *cPtrNullNode.output(0),
      *cPtrNullNode.output(0));

  auto & x1 = GraphExport::Create(*ptrCmpNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*ptrCmpNode2.output(0), "x2");
  auto & x3 = GraphExport::Create(*ptrCmpNode3.output(0), "x3");
  auto & x4 = GraphExport::Create(*ptrCmpNode4.output(0), "x4");
  auto & x5 = GraphExport::Create(*ptrCmpNode5.output(0), "x5");
  auto & x6 = GraphExport::Create(*ptrCmpNode6.output(0), "x6");
  auto & x7 = GraphExport::Create(*ptrCmpNode7.output(0), "x7");

  // Act
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode1);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode2);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode3);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode4);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode5);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode6);
  rvsdg::ReduceNode<PtrCmpOperation>(PtrCmpOperation::normalizeNullPointerComparison, ptrCmpNode7);

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

  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x6.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1u);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 1u);
  }

  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x7.origin());
    EXPECT_NE(constantOperation, nullptr);
    EXPECT_EQ(constantOperation->Representation().nbits(), 1u);
    EXPECT_EQ(constantOperation->Representation().to_uint(), 0u);
  }
}

TEST(FCmpOperationTests, testFoldConstants)
{
  using namespace jlm::rvsdg;

  Graph graph;
  auto region = &graph.GetRootRegion();

  auto fpt = FloatingPointType::Create(fpsize::dbl);

  {
    auto & cNeg1 = ConstantFP::createNode(*region, fpsize::dbl, ::llvm::APFloat(-1.0));
    auto & c0 = ConstantFP::createNode(*region, fpsize::dbl, ::llvm::APFloat(0.0));
    auto & c1 = ConstantFP::createNode(*region, fpsize::dbl, ::llvm::APFloat(1.0));
    auto & c2 = ConstantFP::createNode(*region, fpsize::dbl, ::llvm::APFloat(2.0));
    auto & cNaN = ConstantFP::createNode(
        *region,
        fpsize::dbl,
        ::llvm::APFloat::getNaN(::llvm::APFloat::IEEEdouble()));

    const auto expectFoldedTo =
        [&](fpcmp predicate, rvsdg::Output * lhs, rvsdg::Output * rhs, bool expected)
    {
      const FCmpOperation operation(predicate, fpt);
      const auto folded = FCmpOperation::foldConstants(operation, { lhs, rhs });
      ASSERT_TRUE(folded.has_value())
          << "Expected folding for predicate " << static_cast<int>(predicate);
      ASSERT_EQ(folded->size(), 1u);

      auto [node, constantOperation] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*(*folded)[0]);
      ASSERT_NE(constantOperation, nullptr);
      EXPECT_EQ(constantOperation->Representation().nbits(), 1u);
      EXPECT_EQ(constantOperation->Representation().to_uint(), expected ? 1u : 0u)
          << "Predicate " << static_cast<int>(predicate);
    };

    // FALSE / TRUE
    expectFoldedTo(fpcmp::FALSE, c0.output(0), c1.output(0), false);
    expectFoldedTo(fpcmp::TRUE, c0.output(0), c1.output(0), true);

    // Ordered comparisons (no NaN)
    expectFoldedTo(fpcmp::oeq, c1.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::oeq, c1.output(0), c2.output(0), false);

    expectFoldedTo(fpcmp::ogt, c2.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::ogt, c1.output(0), c2.output(0), false);

    expectFoldedTo(fpcmp::oge, c2.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::oge, c1.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::oge, c1.output(0), c2.output(0), false);

    expectFoldedTo(fpcmp::olt, c1.output(0), c2.output(0), true);
    expectFoldedTo(fpcmp::olt, c2.output(0), c1.output(0), false);

    expectFoldedTo(fpcmp::ole, c1.output(0), c2.output(0), true);
    expectFoldedTo(fpcmp::ole, c1.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::ole, c2.output(0), c1.output(0), false);

    expectFoldedTo(fpcmp::one, c1.output(0), c2.output(0), true);
    expectFoldedTo(fpcmp::one, c1.output(0), c1.output(0), false);

    expectFoldedTo(fpcmp::ord, cNeg1.output(0), c2.output(0), true);
    expectFoldedTo(fpcmp::uno, cNeg1.output(0), c2.output(0), false);

    // Unordered comparisons (NaN)
    expectFoldedTo(fpcmp::ord, cNaN.output(0), c1.output(0), false);
    expectFoldedTo(fpcmp::uno, cNaN.output(0), c1.output(0), true);

    expectFoldedTo(fpcmp::ueq, cNaN.output(0), cNaN.output(0), true);
    expectFoldedTo(fpcmp::ueq, c1.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::ueq, c1.output(0), c2.output(0), false);

    expectFoldedTo(fpcmp::une, cNaN.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::une, c1.output(0), c1.output(0), false);
    expectFoldedTo(fpcmp::une, c1.output(0), c2.output(0), true);

    expectFoldedTo(fpcmp::ugt, cNaN.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::ugt, c2.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::ugt, c1.output(0), c2.output(0), false);

    expectFoldedTo(fpcmp::uge, cNaN.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::uge, c2.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::uge, c1.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::uge, c1.output(0), c2.output(0), false);

    expectFoldedTo(fpcmp::ult, cNaN.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::ult, c1.output(0), c2.output(0), true);
    expectFoldedTo(fpcmp::ult, c2.output(0), c1.output(0), false);

    expectFoldedTo(fpcmp::ule, cNaN.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::ule, c1.output(0), c2.output(0), true);
    expectFoldedTo(fpcmp::ule, c1.output(0), c1.output(0), true);
    expectFoldedTo(fpcmp::ule, c2.output(0), c1.output(0), false);
  }

  {
    auto & nonConst = LlvmGraphImport::create(
        graph,
        fpt,
        fpt,
        "x",
        Linkage::externalLinkage,
        CallingConvention::C,
        false,
        8);

    auto & c0Node = ConstantFP::createNode(*region, fpsize::dbl, ::llvm::APFloat(0.0));

    const FCmpOperation operation(fpcmp::oeq, fpt);
    const auto notFolded = FCmpOperation::foldConstants(
        operation,
        std::vector<rvsdg::Output *>({ &nonConst, c0Node.output(0) }));
    EXPECT_FALSE(notFolded.has_value());
  }
}

}
