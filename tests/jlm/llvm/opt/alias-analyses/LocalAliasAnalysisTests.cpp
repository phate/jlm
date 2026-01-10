/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/opt/alias-analyses/LocalAliasAnalysis.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>

/**
 * Helper function for expecting an alias query to return a given result
 */
static void
Expect(
    jlm::llvm::aa::AliasAnalysis & aa,
    const jlm::rvsdg::Output & p1,
    size_t s1,
    const jlm::rvsdg::Output & p2,
    size_t s2,
    jlm::llvm::aa::AliasAnalysis::AliasQueryResponse expected)
{
  const auto actual = aa.Query(p1, s1, p2, s2);
  EXPECT_EQ(actual, expected);

  // An alias analysis query should always be symmetrical, so check the opposite as well
  const auto mirror = aa.Query(p2, s2, p1, s1);
  EXPECT_EQ(mirror, expected);
}

/**
 * This class sets up an RVSDG representing the following code:
 *
 * \code{.c}
 *   char* getPtr();
 *
 *   extern int global;
 *   extern short globalShort;
 *   extern int array[10];
 *
 *   void func(int** p) {
 *     int alloca1, alloca2;
 *
 *     // Calculate the same offset in two ways
 *     int* q = *p + 2;
 *     int* qPlus2 = *p + 4;
 *     int* qAgain = qPlus2 - 2;
 *
 *     // Different offsets into storage instances
 *     int* arr1 = array + 1;
 *     int* arr2 = array + 2;
 *     int* arr3 = array + 3;
 *     int* arrUnknown = array + global;
 *
 *     // Make alloca2 "escape" the function
 *     *p = &alloca2;
 *
 *     char* bytePtr = getPtr();
 *     char* bytePtrPlus2 = bytePtr + 2;
 *
 *     // All alias queries happen here
 *   }
 * \endcode
 */
class LocalAliasAnalysisTest1 final : public jlm::llvm::RvsdgTest
{
  struct Outputs
  {
    jlm::rvsdg::Output * GetPtr = {};
    jlm::rvsdg::Output * Global = {};
    jlm::rvsdg::Output * GlobalShort = {};
    jlm::rvsdg::Output * Array = {};
    jlm::rvsdg::Output * Func = {};
    jlm::rvsdg::Output * P = {};
    jlm::rvsdg::Output * Alloca1 = {};
    jlm::rvsdg::Output * Alloca2 = {};
    jlm::rvsdg::Output * Q = {};
    jlm::rvsdg::Output * QPlus2 = {};
    jlm::rvsdg::Output * QAgain = {};
    jlm::rvsdg::Output * Arr1 = {};
    jlm::rvsdg::Output * Arr2 = {};
    jlm::rvsdg::Output * Arr3 = {};
    jlm::rvsdg::Output * ArrUnknown = {};
    jlm::rvsdg::Output * BytePtr = {};
    jlm::rvsdg::Output * BytePtrPlus2 = {};
  };

public:
  const Outputs &
  GetOutputs() const noexcept
  {
    return Outputs_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override
  {
    using namespace jlm;
    using namespace jlm::llvm;

    auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto & rvsdg = rvsdgModule->Rvsdg();

    const auto pointerType = PointerType::Create();
    const auto intType = rvsdg::BitType::Create(32);
    const auto shortType = rvsdg::BitType::Create(16);
    const auto byteType = rvsdg::BitType::Create(8);
    const auto intArrayType = ArrayType::Create(intType, 10);
    const auto ioStateType = IOStateType::Create();
    const auto memoryStateType = MemoryStateType::Create();

    const auto funcType = rvsdg::FunctionType::Create(
        { pointerType, ioStateType, memoryStateType },
        { ioStateType, memoryStateType });

    const auto getPtrFuncType = rvsdg::FunctionType::Create(
        { ioStateType, memoryStateType },
        { pointerType, ioStateType, memoryStateType });

    Outputs_.GetPtr = &GraphImport::Create(
        rvsdg,
        getPtrFuncType,
        getPtrFuncType,
        "getPtr",
        Linkage::externalLinkage);

    Outputs_.Global =
        &GraphImport::Create(rvsdg, intType, pointerType, "global", Linkage::externalLinkage);
    Outputs_.GlobalShort = &GraphImport::Create(
        rvsdg,
        shortType,
        pointerType,
        "globalShort",
        Linkage::externalLinkage);
    Outputs_.Array =
        &GraphImport::Create(rvsdg, intArrayType, pointerType, "array", Linkage::externalLinkage);

    // Setup the function "func"
    {
      auto & lambdaNode = *rvsdg::LambdaNode::Create(
          rvsdg.GetRootRegion(),
          LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

      Outputs_.P = lambdaNode.GetFunctionArguments()[0];
      auto ioState = lambdaNode.GetFunctionArguments()[1];
      auto memoryState = lambdaNode.GetFunctionArguments()[2];

      const auto getPtrCtxVar = lambdaNode.AddContextVar(*Outputs_.GetPtr).inner;
      const auto arrayCtxVar = lambdaNode.AddContextVar(*Outputs_.Array).inner;
      const auto globalCtxVar = lambdaNode.AddContextVar(*Outputs_.Global).inner;

      const auto constantOne =
          &rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, 1 });
      const auto constantTwo =
          &rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, 2 });
      const auto constantThree =
          &rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, 3 });
      const auto constantFour =
          &rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, 4 });
      const auto constantMinusTwo =
          &rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, -2 });

      const auto alloca1Outputs = AllocaOperation::create(intType, constantOne, 4);
      const auto alloca2Outputs = AllocaOperation::create(intType, constantOne, 4);

      Outputs_.Alloca1 = alloca1Outputs[0];
      Outputs_.Alloca2 = alloca2Outputs[0];

      memoryState =
          MemoryStateMergeOperation::Create({ memoryState, alloca1Outputs[1], alloca2Outputs[1] });

      // Load from the pointer p
      const auto loadP =
          LoadNonVolatileOperation::Create(Outputs_.P, { memoryState }, pointerType, 8);
      memoryState = loadP[1];

      Outputs_.Q = GetElementPtrOperation::Create(loadP[0], { constantTwo }, intType, pointerType);
      Outputs_.QPlus2 =
          GetElementPtrOperation::Create(loadP[0], { constantFour }, intType, pointerType);
      Outputs_.QAgain = GetElementPtrOperation::Create(
          Outputs_.QPlus2,
          { constantMinusTwo },
          intType,
          pointerType);

      // Create offsets into array
      Outputs_.Arr1 =
          GetElementPtrOperation::Create(arrayCtxVar, { constantOne }, intType, pointerType);
      Outputs_.Arr2 =
          GetElementPtrOperation::Create(arrayCtxVar, { constantTwo }, intType, pointerType);
      Outputs_.Arr3 =
          GetElementPtrOperation::Create(arrayCtxVar, { constantThree }, intType, pointerType);

      // Create a load of the global integer variable "global"
      const auto loadGlobal =
          LoadNonVolatileOperation::Create(globalCtxVar, { memoryState }, intType, 4);
      memoryState = loadGlobal[1];
      Outputs_.ArrUnknown =
          GetElementPtrOperation::Create(arrayCtxVar, { loadGlobal[0] }, byteType, pointerType);

      // Make alloca2 escape
      const auto storeOutputs =
          StoreNonVolatileOperation::Create(Outputs_.P, Outputs_.Alloca2, { memoryState }, 4);
      memoryState = storeOutputs[0];

      // Get bytePtr by calling getPtr()
      const auto callOutputs =
          CallOperation::Create(getPtrCtxVar, getPtrFuncType, { ioState, memoryState });
      Outputs_.BytePtr = callOutputs[0];
      ioState = callOutputs[1];
      memoryState = callOutputs[2];

      Outputs_.BytePtrPlus2 =
          GetElementPtrOperation::Create(Outputs_.BytePtr, { constantTwo }, byteType, pointerType);

      lambdaNode.finalize({ ioState, memoryState });
      Outputs_.Func = lambdaNode.output();
    }

    return rvsdgModule;
  }

  Outputs Outputs_ = {};
};

TEST(LocalAliasAnalysisTests, TestLocalAliasAnalysis)
{
  using namespace jlm::llvm::aa;

  // Arrange
  LocalAliasAnalysisTest1 rvsdg;
  rvsdg.InitializeTest();
  const auto & outputs = rvsdg.GetOutputs();

  jlm::rvsdg::view(&rvsdg.graph().GetRootRegion(), stdout);

  LocalAliasAnalysis aa;

  // Assert

  // Distinct global variables do not alias
  Expect(aa, *outputs.Global, 4, *outputs.GlobalShort, 2, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Global, 4, *outputs.Arr1, 4, AliasAnalysis::NoAlias);

  // An alloca never aliases any other memory allocating operation
  Expect(aa, *outputs.Alloca2, 4, *outputs.Alloca1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 4, *outputs.Global, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 4, *outputs.Array, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 4, *outputs.Arr1, 4, AliasAnalysis::NoAlias);

  // An alloca that has not "escaped" can not alias external pointers
  Expect(aa, *outputs.Alloca1, 4, *outputs.BytePtr, 4, AliasAnalysis::NoAlias);

  // An alloca that has "escaped" may alias external pointers
  Expect(aa, *outputs.Alloca2, 4, *outputs.BytePtr, 4, AliasAnalysis::MayAlias);

  // Distinct offsets can not alias, unless the access regions overlap
  Expect(aa, *outputs.Q, 8, *outputs.QPlus2, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Q, 9, *outputs.QPlus2, 8, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Q, 8, *outputs.QPlus2, 16, AliasAnalysis::NoAlias);

  // Identical offsets are MustAlias
  Expect(aa, *outputs.Q, 4, *outputs.QAgain, 4, AliasAnalysis::MustAlias);

  // q is at least 8 bytes into the storage instance of *p
  // so it can not alias with the first 8 bytes of array
  Expect(aa, *outputs.Array, 8, *outputs.Q, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Array, 9, *outputs.Q, 4, AliasAnalysis::MayAlias);
  // We know that arr1, arr2 and arr3 are 4, 8 and 12 bytes into array
  Expect(aa, *outputs.Arr1, 4, *outputs.Q, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Arr1, 5, *outputs.Q, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Arr2, 4, *outputs.Q, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Arr3, 4, *outputs.Q, 4, AliasAnalysis::MayAlias);

  // An unknown offset into array can only alias with array, at all offsets
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Array, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Arr1, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Arr2, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Arr3, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Global, 4, AliasAnalysis::NoAlias);
  // Q may be a pointer into array, so it is also "MayAlias"
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Q, 4, AliasAnalysis::MayAlias);

  // We know that q is at least 16 bytes into its storage instance,
  // so it may not alias with storage instances that are 16 bytes or less
  Expect(aa, *outputs.Q, 4, *outputs.Global, 4, AliasAnalysis::NoAlias);

  // A five byte operation can never target the 4 byte global variable
  Expect(aa, *outputs.BytePtr, 5, *outputs.Global, 4, AliasAnalysis::NoAlias);
  // A four byte operation can, however
  Expect(aa, *outputs.BytePtr, 4, *outputs.Global, 4, AliasAnalysis::MayAlias);
  // Even a 40 byte operation can target the 40 byte global array
  Expect(aa, *outputs.BytePtr, 40, *outputs.Array, 4, AliasAnalysis::MayAlias);
  // The 40 byte operation may overlap with 4 bytes at any offset within the Array
  Expect(aa, *outputs.BytePtr, 40, *outputs.Arr3, 4, AliasAnalysis::MayAlias);

  // BytePtrPlus2 has an offset of at least 2, so can not alias with the first 2 bytes of anything
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Alloca2, 2, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Alloca2, 3, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Array, 2, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Array, 3, AliasAnalysis::MayAlias);
  // Arr1 is already 4 bytes into array, so BytePtrPlus2 can alias with it
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Arr1, 2, AliasAnalysis::MayAlias);
}

/**
 * This class sets up an RVSDG representing the following code:
 *
 * \code{.c}
 *
 *   void func(int1_t x, int32_t* ptr) {
 *     int32_t alloca1
 *     int64_t alloca2;
 *     int32_t alloca3[2];
 *
 *     // Using a gamma node
 *     int32_t* allocaUnknown;
 *     if (x)
 *       allocaUnknown = &alloca1;
 *     else
 *       allocaUnknown = (int32_t*) &alloca2;
 *
 *     int32_t* allocaUnknownPlus1 = allocaUnknown + 1;
 *     int32_t* alloca3Plus1 = alloca3 + 1;
 *
 *     // Using a select operation
 *     int32_t* alloca3UnknownOffset = x ? alloca3 : alloca3Plus1;
 *
 *     // Using a select operation that is actually a nop
 *     int32_t* alloca3KnownOffset = x ? alloca3Plus1 : alloca3Plus1;
 *
 *     // All alias queries happen here
 *   }
 * \endcode
 */
class LocalAliasAnalysisTest2 final : public jlm::llvm::RvsdgTest
{
  struct Outputs
  {
    jlm::rvsdg::Output * Func = {};
    jlm::rvsdg::Output * X = {};
    jlm::rvsdg::Output * Ptr = {};
    jlm::rvsdg::Output * Alloca1 = {};
    jlm::rvsdg::Output * Alloca2 = {};
    jlm::rvsdg::Output * Alloca3 = {};
    jlm::rvsdg::Output * AllocaUnknown = {};
    jlm::rvsdg::Output * AllocaUnknownPlus1 = {};
    jlm::rvsdg::Output * Alloca3Plus1 = {};
    jlm::rvsdg::Output * Alloca3UnknownOffset = {};
    jlm::rvsdg::Output * Alloca3KnownOffset = {};
  };

public:
  const Outputs &
  GetOutputs() const noexcept
  {
    return Outputs_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override
  {
    using namespace jlm;
    using namespace jlm::llvm;

    auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto & rvsdg = rvsdgModule->Rvsdg();

    const auto pointerType = PointerType::Create();
    const auto int1Type = rvsdg::BitType::Create(1);
    const auto int32Type = rvsdg::BitType::Create(32);
    const auto int64Type = rvsdg::BitType::Create(64);
    const auto intArrayType = ArrayType::Create(int32Type, 2);
    const auto ioStateType = IOStateType::Create();
    const auto memoryStateType = MemoryStateType::Create();

    const auto funcType = rvsdg::FunctionType::Create(
        { int1Type, pointerType, ioStateType, memoryStateType },
        { ioStateType, memoryStateType });

    // Setup the function "func"
    {
      auto & lambdaNode = *rvsdg::LambdaNode::Create(
          rvsdg.GetRootRegion(),
          LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

      Outputs_.X = lambdaNode.GetFunctionArguments()[0];
      Outputs_.Ptr = lambdaNode.GetFunctionArguments()[1];
      auto ioState = lambdaNode.GetFunctionArguments()[2];
      auto memoryState = lambdaNode.GetFunctionArguments()[3];

      const auto constantZero =
          &rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, 0 });
      const auto constantOne =
          &rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, 1 });

      const auto alloca1Outputs = AllocaOperation::create(int32Type, constantOne, 4);
      const auto alloca2Outputs = AllocaOperation::create(int64Type, constantOne, 4);
      const auto alloca3Outputs = AllocaOperation::create(intArrayType, constantOne, 4);

      Outputs_.Alloca1 = alloca1Outputs[0];
      Outputs_.Alloca2 = alloca2Outputs[0];
      Outputs_.Alloca3 = alloca3Outputs[0];

      memoryState = MemoryStateMergeOperation::Create(
          { memoryState, alloca1Outputs[1], alloca2Outputs[1], alloca3Outputs[1] });

      const auto matchResult = rvsdg::MatchOperation::Create(*Outputs_.X, { { 1, 1 } }, 0, 2);
      const auto gamma = rvsdg::GammaNode::create(matchResult, 2);
      const auto entryVarA1 = gamma->AddEntryVar(Outputs_.Alloca1);
      const auto entryVarA2 = gamma->AddEntryVar(Outputs_.Alloca2);
      const auto exitVar =
          gamma->AddExitVar({ entryVarA1.branchArgument[0], entryVarA2.branchArgument[1] });
      Outputs_.AllocaUnknown = exitVar.output;

      Outputs_.AllocaUnknownPlus1 = GetElementPtrOperation::Create(
          Outputs_.AllocaUnknown,
          { constantOne },
          int32Type,
          pointerType);

      Outputs_.Alloca3Plus1 = GetElementPtrOperation::Create(
          Outputs_.Alloca3,
          { constantZero, constantOne },
          intArrayType,
          pointerType);

      Outputs_.Alloca3UnknownOffset = rvsdg::CreateOpNode<SelectOperation>(
                                          { Outputs_.X, Outputs_.Alloca3, Outputs_.Alloca3Plus1 },
                                          pointerType)
                                          .output(0);

      Outputs_.Alloca3KnownOffset =
          rvsdg::CreateOpNode<SelectOperation>(
              { Outputs_.X, Outputs_.Alloca3Plus1, Outputs_.Alloca3Plus1 },
              pointerType)
              .output(0);

      lambdaNode.finalize({ ioState, memoryState });
      Outputs_.Func = lambdaNode.output();
    }

    return rvsdgModule;
  }

  Outputs Outputs_ = {};
};

TEST(LocalAliasAnalysisTests, TestLocalAliasAnalysisMultipleOrigins)
{
  using namespace jlm::llvm::aa;

  // Arrange
  LocalAliasAnalysisTest2 rvsdg;
  rvsdg.InitializeTest();
  const auto & outputs = rvsdg.GetOutputs();

  jlm::rvsdg::view(&rvsdg.graph().GetRootRegion(), stdout);

  LocalAliasAnalysis aa;

  // Assert

  // First check that none of the allocas have been mixed up with unknown pointers
  Expect(aa, *outputs.Alloca1, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3Plus1, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);

  // Check that allocaUnknown may alias only alloca1 or alloca2
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Alloca1, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Alloca2, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Alloca3, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Alloca3Plus1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);

  // If performing an 8 byte operation, it may only alias alloca2, becoming a must alias
  Expect(aa, *outputs.AllocaUnknown, 8, *outputs.Alloca1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknown, 8, *outputs.Alloca2, 4, AliasAnalysis::MustAlias);
  // Performing a 9 byte operation is neither legal for alloca1 nor alloca2
  Expect(aa, *outputs.AllocaUnknown, 9, *outputs.Alloca2, 4, AliasAnalysis::NoAlias);

  // Adding a 4 byte offset forces all operations to be on alloca2
  Expect(aa, *outputs.AllocaUnknownPlus1, 1, *outputs.Alloca1, 1, AliasAnalysis::NoAlias);
  // We also know that we are 4 bytes into alloca2
  Expect(aa, *outputs.AllocaUnknownPlus1, 4, *outputs.Alloca2, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknownPlus1, 4, *outputs.Alloca2, 5, AliasAnalysis::MayAlias);
  // Performing a 5 byte operation is neither legal for alloca1 nor alloca2
  Expect(aa, *outputs.AllocaUnknownPlus1, 5, *outputs.Alloca2, 8, AliasAnalysis::NoAlias);

  // Check that the offset of allocaUnknown is correctly calculated (4 bytes)
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.AllocaUnknownPlus1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknown, 5, *outputs.AllocaUnknownPlus1, 4, AliasAnalysis::MayAlias);

  // Check that the pointer with an unknown offset into alloca3 does not alias anything else
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Alloca1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Alloca2, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.AllocaUnknown, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);

  // It may alias alloca3 and alloca3 + 1
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Alloca3, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Alloca3Plus1, 4, AliasAnalysis::MayAlias);

  // If performing an 8 byte operation, we know that we are at the start of alloca3
  Expect(aa, *outputs.Alloca3UnknownOffset, 8, *outputs.Alloca3, 4, AliasAnalysis::MustAlias);
  // We still overlap with the second half of alloca3
  Expect(aa, *outputs.Alloca3UnknownOffset, 8, *outputs.Alloca3Plus1, 4, AliasAnalysis::MayAlias);

  // The select with duplicate operands should be a single origin: alloca3
  Expect(aa, *outputs.Alloca3KnownOffset, 4, *outputs.Alloca3, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3KnownOffset, 4, *outputs.Alloca3Plus1, 4, AliasAnalysis::MustAlias);
}
