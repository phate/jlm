/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/backend/IpGraphToLlvmConverter.hpp>
#include <jlm/llvm/backend/RvsdgToIpGraphConverter.hpp>
#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/util/Statistics.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>

#include <algorithm>

class LlvmBackendCastingFixture : public testing::TestWithParam<int>
{
};

TEST_P(LlvmBackendCastingFixture, AllIntegerCasts)
{
  /*
   * Creates a function equivalent to the following C code in RVSDG:
   *
   * <SIZE x i8> f(uint32_t x) {
   *   uint64_t zext = x;
   *   uint32_t trunc = (uint32_t) zext;
   *   int64_t sext = (int32_t) trunc;
   *   void* inttoptr = (void*) sext;
   *   uint64_t ptrtoint = (uint64_t) inttoptr;
   *   <SIZE x i8> bitcast = (<SIZE x i8>) ptrtoint;     // where SIZE = sizeof(x);
   *   return bitcast;
   * }
   *
   * The test is parameterized with a vectorization width.
   * When it is non-zero, all scalar types are replaced with vectors of the given width.
   * E.g., with a width of 4, all uint64_t values become <4 x uint64_t> instead.
   *
   * The test converts the above RVSDG to LLVM IR, checking that all casts have been converted,
   * with the expected input and output types.
   */

  // If 0, no vectorization is used
  int vectorization = GetParam();

  // Arrange
  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule->Rvsdg();

  const auto bits32 = jlm::rvsdg::BitType::Create(32);
  const auto bits64 = jlm::rvsdg::BitType::Create(64);
  const auto pointerType = jlm::llvm::PointerType::Create();

  // Create the vectorized versions of the above types
  std::shared_ptr<const jlm::rvsdg::Type> bits32V, bits64V, pointerTypeV;
  if (vectorization == 0)
  {
    bits32V = bits32;
    bits64V = bits64;
    pointerTypeV = pointerType;
  }
  else
  {
    bits32V = jlm::llvm::FixedVectorType::Create(bits32, vectorization);
    bits64V = jlm::llvm::FixedVectorType::Create(bits64, vectorization);
    pointerTypeV = jlm::llvm::FixedVectorType::Create(pointerType, vectorization);
  }

  // Create the byte vector used for bitcasting
  const size_t sizeofX = 8 * std::max(1, vectorization);
  const auto byteType = jlm::rvsdg::BitType::Create(8);
  const auto byteVectorType = jlm::llvm::FixedVectorType::Create(byteType, sizeofX);

  auto functionType = jlm::rvsdg::FunctionType::Create({ bits32V }, { byteVectorType });
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          functionType,
          "f",
          jlm::llvm::Linkage::externalLinkage));

  auto x = lambda->GetFunctionArguments()[0];

  // Helper function for wrapping operations in VectorUnaryOperation when vectorization is enabled
  auto createUnary =
      [&](const auto & op,
          jlm::rvsdg::Output & operand,
          const std::shared_ptr<const jlm::rvsdg::Type> & resultType) -> jlm::rvsdg::Output &
  {
    if (vectorization == 0)
    {
      return *jlm::rvsdg::CreateOpNode<std::decay_t<decltype(op)>>(
                  { &operand },
                  op.argument(0),
                  op.result(0))
                  .output(0);
    }

    return *jlm::rvsdg::CreateOpNode<jlm::llvm::VectorUnaryOperation>(
                { &operand },
                op,
                std::static_pointer_cast<const jlm::llvm::VectorType>(operand.Type()),
                std::static_pointer_cast<const jlm::llvm::VectorType>(resultType))
                .output(0);
  };

  // Create the function body
  auto & zext = createUnary(jlm::llvm::ZExtOperation(bits32, bits64), *x, bits64V);
  auto & trunc = createUnary(jlm::llvm::TruncOperation(bits64, bits32), zext, bits32V);
  auto & sext = createUnary(jlm::llvm::SExtOperation(bits32, bits64), trunc, bits64V);
  auto & inttoptr =
      createUnary(jlm::llvm::IntegerToPointerOperation(bits64, pointerType), sext, pointerTypeV);
  auto & ptrtoint =
      createUnary(jlm::llvm::PtrToIntOperation(pointerType, bits64), inttoptr, bits64V);
  // Bitcasts are never wrapped in VectorUnaryOperation
  auto & bitcast = *jlm::llvm::BitCastOperation::create(&ptrtoint, byteVectorType);

  auto lambdaOutput = lambda->finalize({ &bitcast });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  auto ipgModule =
      jlm::llvm::RvsdgToIpGraphConverter::CreateAndConvertModule(*rvsdgModule, statisticsCollector);
  llvm::LLVMContext context;
  auto llvmModule = jlm::llvm::IpGraphToLlvmConverter::CreateAndConvertModule(*ipgModule, context);

  // Assert
  {
    llvm::Type * expectedBits32 = llvm::Type::getInt32Ty(context);
    llvm::Type * expectedBits64 = llvm::Type::getInt64Ty(context);
    llvm::Type * expectedPointer = llvm::PointerType::getUnqual(context);

    if (vectorization != 0)
    {
      auto ec = llvm::ElementCount::getFixed(vectorization);
      expectedBits32 = llvm::VectorType::get(expectedBits32, ec);
      expectedBits64 = llvm::VectorType::get(expectedBits64, ec);
      expectedPointer = llvm::VectorType::get(expectedPointer, ec);
    }

    auto byteEc = llvm::ElementCount::getFixed(sizeofX);
    llvm::Type * expectedByteVector = llvm::VectorType::get(llvm::Type::getInt8Ty(context), byteEc);

    auto llvmFunction = llvmModule->getFunction("f");
    ASSERT_NE(llvmFunction, nullptr);
    EXPECT_EQ(llvmFunction->getReturnType(), expectedByteVector);
    ASSERT_EQ(llvmFunction->arg_size(), 1u);
    EXPECT_EQ(llvmFunction->arg_begin()->getType(), expectedBits32);

    size_t numZext = 0;
    size_t numTrunc = 0;
    size_t numSext = 0;
    size_t numInttoptr = 0;
    size_t numPtrtoint = 0;
    size_t numBitcasts = 0;

    for (auto & basicBlock : *llvmFunction)
    {
      for (auto & instruction : basicBlock)
      {
        if (auto * zextInstruction = llvm::dyn_cast<llvm::ZExtInst>(&instruction))
        {
          numZext++;
          EXPECT_EQ(zextInstruction->getSrcTy(), expectedBits32);
          EXPECT_EQ(zextInstruction->getDestTy(), expectedBits64);
        }
        else if (auto * truncInstruction = llvm::dyn_cast<llvm::TruncInst>(&instruction))
        {
          numTrunc++;
          EXPECT_EQ(truncInstruction->getSrcTy(), expectedBits64);
          EXPECT_EQ(truncInstruction->getDestTy(), expectedBits32);
        }
        else if (auto * sextInstruction = llvm::dyn_cast<llvm::SExtInst>(&instruction))
        {
          numSext++;
          EXPECT_EQ(sextInstruction->getSrcTy(), expectedBits32);
          EXPECT_EQ(sextInstruction->getDestTy(), expectedBits64);
        }
        else if (auto * intToPtrInstruction = llvm::dyn_cast<llvm::IntToPtrInst>(&instruction))
        {
          numInttoptr++;
          EXPECT_EQ(intToPtrInstruction->getSrcTy(), expectedBits64);
          EXPECT_EQ(intToPtrInstruction->getDestTy(), expectedPointer);
        }
        else if (auto * ptrToIntInstruction = llvm::dyn_cast<llvm::PtrToIntInst>(&instruction))
        {
          numPtrtoint++;
          EXPECT_EQ(ptrToIntInstruction->getSrcTy(), expectedPointer);
          EXPECT_EQ(ptrToIntInstruction->getDestTy(), expectedBits64);
        }
        else if (auto * bitcastInstruction = llvm::dyn_cast<llvm::BitCastInst>(&instruction))
        {
          numBitcasts++;
          EXPECT_EQ(bitcastInstruction->getSrcTy(), expectedBits64);
          EXPECT_EQ(bitcastInstruction->getDestTy(), expectedByteVector);
        }
      }
    }

    auto * returnInstruction =
        llvm::dyn_cast<llvm::ReturnInst>(llvmFunction->back().getTerminator());
    ASSERT_NE(returnInstruction, nullptr);
    auto * bitcastInstruction =
        llvm::dyn_cast<llvm::BitCastInst>(returnInstruction->getReturnValue());
    EXPECT_NE(bitcastInstruction, nullptr);

    EXPECT_EQ(numZext, 1u);
    EXPECT_EQ(numTrunc, 1u);
    EXPECT_EQ(numSext, 1u);
    EXPECT_EQ(numInttoptr, 1u);
    EXPECT_EQ(numPtrtoint, 1u);
    EXPECT_EQ(numBitcasts, 1u);
  }
}

INSTANTIATE_TEST_SUITE_P(
    LlvmBackendCastingTests,
    LlvmBackendCastingFixture,
    testing::Values(0, 1, 2, 4, 8));
