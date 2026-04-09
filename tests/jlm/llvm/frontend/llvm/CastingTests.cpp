/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/MatchType.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <algorithm>

class LlvmFrontendCastingFixture : public testing::TestWithParam<int>
{
};

TEST_P(LlvmFrontendCastingFixture, AllIntegerCasts)
{
  /**
   * Creates a function equivalent to the following C code in LLVM IR:
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
   */

  // If 0, no vectorization is used
  int vectorization = GetParam();

  // Arrange
  llvm::LLVMContext context;
  llvm::Module llvmModule("module", context);

  llvm::Type * int64Type = llvm::Type::getInt64Ty(context);
  llvm::Type * int32Type = llvm::Type::getInt32Ty(context);
  llvm::Type * pointerType = llvm::PointerType::getUnqual(context);

  // If vectorization is enabled, make all the types vectors
  if (vectorization != 0)
  {
    auto ec = llvm::ElementCount::getFixed(vectorization);
    int64Type = llvm::VectorType::get(int64Type, ec);
    int32Type = llvm::VectorType::get(int32Type, ec);
    pointerType = llvm::VectorType::get(pointerType, ec);
  }

  // Create the final <SIZE x i8> type
  size_t sizeofX = 8 * std::max(1, vectorization);
  auto ec = llvm::ElementCount::getFixed(sizeofX);
  llvm::Type * byteVectorType = llvm::VectorType::get(llvm::Type::getInt8Ty(context), ec);

  auto functionType =
      llvm::FunctionType::get(byteVectorType, llvm::ArrayRef<llvm::Type *>({ int32Type }), false);
  auto function =
      llvm::Function::Create(functionType, llvm::GlobalValue::ExternalLinkage, "f", &llvmModule);

  auto basicBlock = llvm::BasicBlock::Create(context, "bb0", function);

  llvm::IRBuilder<> builder(basicBlock);
  auto zext = builder.CreateZExt(function->arg_begin(), int64Type);
  auto trunc = builder.CreateTrunc(zext, int32Type);
  auto sext = builder.CreateSExt(trunc, int64Type);
  auto inttoptr = builder.CreateIntToPtr(sext, pointerType);
  auto ptrtoint = builder.CreatePtrToInt(inttoptr, int64Type);
  auto bitcast = builder.CreateBitCast(ptrtoint, byteVectorType);
  builder.CreateRet(bitcast);

  llvmModule.print(llvm::errs(), nullptr);

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(llvmModule);
  print(*ipgModule, stdout);

  // Assert
  {
    using namespace jlm::llvm;

    const auto jlmBits32 = jlm::rvsdg::BitType::Create(32);
    const auto jlmBits64 = jlm::rvsdg::BitType::Create(64);
    const auto jlmPointerType = jlm::llvm::PointerType::Create();
    const auto jlmByteType = jlm::rvsdg::BitType::Create(8);
    const auto jlmByteVectorType = jlm::llvm::FixedVectorType::Create(jlmByteType, sizeofX);

    auto controlFlowGraph =
        dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("f"))->cfg();
    auto basicBlock =
        dynamic_cast<const jlm::llvm::BasicBlock *>(controlFlowGraph->entry()->OutEdge(0)->sink());

    size_t numUnaryVector = 0;
    size_t numZext = 0;
    size_t numTrunc = 0;
    size_t numSext = 0;
    size_t numInttoptr = 0;
    size_t numPtrtoint = 0;
    size_t numBitcasts = 0;
    for (auto it = basicBlock->begin(); it != basicBlock->end(); it++)
    {
      auto op = &(*it)->operation();

      // If the operation is wrapped in a vector unary, unwrap it
      if (auto vecOp = dynamic_cast<const VectorUnaryOperation *>(op))
      {
        numUnaryVector++;
        op = &vecOp->operation();
      }

      std::cout << op->debug_string() << std::endl;
      jlm::rvsdg::MatchTypeOrFail(
          *op,
          [&]([[maybe_unused]] const UndefValueOperation & op)
          {
            // Ignore the undef operation created as a default return value
          },
          [&]([[maybe_unused]] const AssignmentOperation & op)
          {
            // Ignore the assignment to the dummy return variable
          },
          [&](const ZExtOperation & op)
          {
            numZext++;
            EXPECT_EQ(*op.argument(0), *jlmBits32);
            EXPECT_EQ(*op.result(0), *jlmBits64);
          },
          [&](const TruncOperation & op)
          {
            numTrunc++;
            EXPECT_EQ(*op.argument(0), *jlmBits64);
            EXPECT_EQ(*op.result(0), *jlmBits32);
          },
          [&](const SExtOperation & op)
          {
            numSext++;
            EXPECT_EQ(*op.argument(0), *jlmBits32);
            EXPECT_EQ(*op.result(0), *jlmBits64);
          },
          [&](const IntegerToPointerOperation & op)
          {
            numInttoptr++;
            EXPECT_EQ(*op.argument(0), *jlmBits64);
            EXPECT_EQ(*op.result(0), *jlmPointerType);
          },
          [&](const PtrToIntOperation & op)
          {
            numPtrtoint++;
            EXPECT_EQ(*op.argument(0), *jlmPointerType);
            EXPECT_EQ(*op.result(0), *jlmBits64);
          },
          [&](const BitCastOperation & op)
          {
            numBitcasts++;

            // BitCasts should never be wrapped in VectorUnaryOperation, so for vectorized
            // instances of the test we expect the operation to take a vector type.
            if (vectorization)
            {
              const auto jlmBits64Vector =
                  jlm::llvm::FixedVectorType::Create(jlmBits64, vectorization);
              EXPECT_EQ(*op.argument(0), *jlmBits64Vector);
            }
            else
            {
              // For the non-vectorized instance of this test, the input is just a uint64
              EXPECT_EQ(*op.argument(0), *jlmBits64);
            }
            EXPECT_EQ(*op.result(0), *jlmByteVectorType);
          });
    }

    EXPECT_EQ(numUnaryVector, vectorization ? 5 : 0u);
    EXPECT_EQ(numZext, 1u);
    EXPECT_EQ(numTrunc, 1u);
    EXPECT_EQ(numSext, 1u);
    EXPECT_EQ(numInttoptr, 1u);
    EXPECT_EQ(numPtrtoint, 1u);
    EXPECT_EQ(numBitcasts, 1u);
  }
}

INSTANTIATE_TEST_SUITE_P(
    LlvmFrontendCastingTests,
    LlvmFrontendCastingFixture,
    testing::Values(0, 1, 2, 4, 8));
