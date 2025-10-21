/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraphAliasAnalysis.hpp>
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
  assert(actual == expected);

  // An alias analysis query should always be symmetrical, so check the opposite as well
  const auto mirror = aa.Query(p2, s2, p1, s1);
  assert(mirror == expected);
}

/**
 * This class sets up an RVSDG representing the following code:
 *
 * \code{.c}
 *   int* global = nullptr;
 *   static int* local = nullptr;
 *   extern int* imported;
 *
 *   int* getPtr();
 *
 *   void func(int** p) {
 *     int alloca1, alloca2, alloca3, alloca4;
 *
 *     int* q = *p;
 *     *p = &alloca1;
 *
 *     // Load global values into virtual registers
 *     int* globalLoad = global;
 *     int* localLoad = local;
 *     int* importedLoad = imported;
 *
 *     // Store to global values
 *     global = &alloca2;
 *     local = &alloca3;
 *     imported = &alloca4;
 *
 *     int* r = getPtr();
 *
 *     // All alias queries happen here
 *   }
 * \endcode
 */
class PtGAliasAnalysisTest final : public jlm::tests::RvsdgTest
{
  struct Outputs
  {
    jlm::rvsdg::Output * Global = {};
    jlm::rvsdg::Output * Local = {};
    jlm::rvsdg::Output * Imported = {};
    jlm::rvsdg::Output * GetPtr = {};
    jlm::rvsdg::Output * Func = {};
    jlm::rvsdg::Output * P = {};
    jlm::rvsdg::Output * Alloca1 = {};
    jlm::rvsdg::Output * Alloca2 = {};
    jlm::rvsdg::Output * Alloca3 = {};
    jlm::rvsdg::Output * Alloca4 = {};
    jlm::rvsdg::Output * Q = {};
    jlm::rvsdg::Output * GlobalLoad = {};
    jlm::rvsdg::Output * LocalLoad = {};
    jlm::rvsdg::Output * ImportedLoad = {};
    jlm::rvsdg::Output * R = {};
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

    // Create the global pointer variable "global", that is exported
    auto & globalDelta = *rvsdg::DeltaNode::Create(
        &rvsdg.GetRootRegion(),
        DeltaOperation::Create(pointerType, "global", Linkage::externalLinkage, "", false));
    {
      const auto nullPtr =
          ConstantPointerNullOperation::Create(globalDelta.subregion(), pointerType);
      globalDelta.finalize(nullPtr);
    }
    rvsdg::GraphExport::Create(globalDelta.output(), "global");
    Outputs_.Global = &globalDelta.output();

    // Create the global variable "local", that is not exported
    auto & localDelta = *rvsdg::DeltaNode::Create(
        &rvsdg.GetRootRegion(),
        DeltaOperation::Create(pointerType, "local", Linkage::internalLinkage, "", false));
    {
      const auto nullPtr =
          ConstantPointerNullOperation::Create(localDelta.subregion(), pointerType);
      localDelta.finalize(nullPtr);
    }
    Outputs_.Local = &localDelta.output();

    Outputs_.Imported =
        &GraphImport::Create(rvsdg, pointerType, pointerType, "imported", Linkage::externalLinkage);

    // Setup the function "func"
    {
      auto & lambdaNode = *rvsdg::LambdaNode::Create(
          rvsdg.GetRootRegion(),
          LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

      Outputs_.P = lambdaNode.GetFunctionArguments()[0];
      auto ioState = lambdaNode.GetFunctionArguments()[1];
      auto memoryState = lambdaNode.GetFunctionArguments()[2];

      const auto getPtrCtxVar = lambdaNode.AddContextVar(*Outputs_.GetPtr).inner;
      const auto globalCtxVar = lambdaNode.AddContextVar(*Outputs_.Global).inner;
      const auto localCtxVar = lambdaNode.AddContextVar(*Outputs_.Local).inner;
      const auto importedCtxVar = lambdaNode.AddContextVar(*Outputs_.Imported).inner;

      const auto constantOne = create_bitconstant(lambdaNode.subregion(), 32, 1);

      const auto alloca1Outputs = AllocaOperation::create(intType, constantOne, 4);
      const auto alloca2Outputs = AllocaOperation::create(intType, constantOne, 4);
      const auto alloca3Outputs = AllocaOperation::create(intType, constantOne, 4);
      const auto alloca4Outputs = AllocaOperation::create(intType, constantOne, 4);

      Outputs_.Alloca1 = alloca1Outputs[0];
      Outputs_.Alloca2 = alloca2Outputs[0];
      Outputs_.Alloca3 = alloca3Outputs[0];
      Outputs_.Alloca4 = alloca4Outputs[0];

      memoryState = MemoryStateMergeOperation::Create({ memoryState,
                                                        alloca1Outputs[1],
                                                        alloca2Outputs[1],
                                                        alloca3Outputs[1],
                                                        alloca4Outputs[1] });

      // Load: q = *p;
      const auto loadP =
          LoadNonVolatileOperation::Create(Outputs_.P, { memoryState }, pointerType, 8);
      Outputs_.Q = loadP[0];
      memoryState = loadP[1];

      // Store the address of alloca1 to p
      const auto storeP =
          StoreNonVolatileOperation::Create(Outputs_.P, Outputs_.Alloca1, { memoryState }, 8);
      memoryState = storeP[0];

      // Create loads of the global variables
      const auto loadGlobal =
          LoadNonVolatileOperation::Create(globalCtxVar, { memoryState }, pointerType, 8);
      Outputs_.GlobalLoad = loadGlobal[0];
      memoryState = loadGlobal[1];

      const auto loadLocal =
          LoadNonVolatileOperation::Create(localCtxVar, { memoryState }, pointerType, 8);
      Outputs_.LocalLoad = loadLocal[0];
      memoryState = loadLocal[1];

      const auto loadImported =
          LoadNonVolatileOperation::Create(importedCtxVar, { memoryState }, pointerType, 8);
      Outputs_.ImportedLoad = loadImported[0];
      memoryState = loadImported[1];

      // Store to global values
      const auto storeGlobal =
          StoreNonVolatileOperation::Create(globalCtxVar, Outputs_.Alloca2, { memoryState }, 8);
      memoryState = storeGlobal[0];

      const auto storeLocal =
          StoreNonVolatileOperation::Create(localCtxVar, Outputs_.Alloca3, { memoryState }, 8);
      memoryState = storeLocal[0];

      const auto storeImported =
          StoreNonVolatileOperation::Create(importedCtxVar, Outputs_.Alloca4, { memoryState }, 8);
      memoryState = storeImported[0];

      // Get r by calling getPtr()
      const auto callOutputs =
          CallOperation::Create(getPtrCtxVar, getPtrFuncType, { ioState, memoryState });
      Outputs_.R = callOutputs[0];
      ioState = callOutputs[1];
      memoryState = callOutputs[2];

      lambdaNode.finalize({ ioState, memoryState });
      Outputs_.Func = lambdaNode.output();
    }
    rvsdg::GraphExport::Create(*Outputs_.Func, "func");

    return rvsdgModule;
  }

  Outputs Outputs_ = {};
};

void
TestPtGAliasAnalysis()
{
  using namespace jlm::llvm::aa;

  // Arrange
  PtGAliasAnalysisTest rvsdg;
  rvsdg.InitializeTest();
  const auto & outputs = rvsdg.GetOutputs();

  // jlm::rvsdg::view(&rvsdg.graph().GetRootRegion(), stdout);

  Andersen andersen;
  auto pointsToGraph = andersen.Analyze(rvsdg.module());
  // std::cout << PointsToGraph::ToDot(*pointsToGraph) << std::endl;
  PointsToGraphAliasAnalysis aa(*pointsToGraph);

  // Assert

  // Distinct global variables do not alias
  Expect(aa, *outputs.Global, 8, *outputs.Local, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Global, 8, *outputs.Imported, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Local, 8, *outputs.Imported, 8, AliasAnalysis::NoAlias);

  // The same global variable aliases itself
  Expect(aa, *outputs.Global, 8, *outputs.Global, 8, AliasAnalysis::MustAlias);
  Expect(aa, *outputs.Local, 8, *outputs.Local, 8, AliasAnalysis::MustAlias);
  Expect(aa, *outputs.Imported, 4, *outputs.Imported, 4, AliasAnalysis::MustAlias);

  // Distinct allocas never alias with each other or globals
  Expect(aa, *outputs.Alloca1, 8, *outputs.Alloca2, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca1, 8, *outputs.Alloca3, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca1, 8, *outputs.Alloca4, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca1, 8, *outputs.Global, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca1, 8, *outputs.Local, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca1, 8, *outputs.Imported, 8, AliasAnalysis::NoAlias);

  // The pointer argument may point to any escaped memory
  Expect(aa, *outputs.P, 8, *outputs.Global, 8, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.P, 8, *outputs.Imported, 8, AliasAnalysis::MayAlias);
  // But not things that have not escaped
  Expect(aa, *outputs.P, 8, *outputs.Local, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.P, 8, *outputs.Alloca3, 8, AliasAnalysis::NoAlias);

  // The pointer q, loaded from p, is likewise unknown
  Expect(aa, *outputs.Q, 8, *outputs.Global, 8, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Q, 8, *outputs.Imported, 8, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Q, 8, *outputs.P, 8, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Q, 8, *outputs.Local, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Q, 8, *outputs.Alloca3, 8, AliasAnalysis::NoAlias);

  // The pointer value loaded from global can point to anything externally available
  Expect(aa, *outputs.GlobalLoad, 4, *outputs.Global, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.GlobalLoad, 4, *outputs.Imported, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.GlobalLoad, 4, *outputs.Alloca1, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.GlobalLoad, 4, *outputs.Local, 4, AliasAnalysis::NoAlias);

  // The pointer value loaded from local can not point to anything (except alloca3)
  Expect(aa, *outputs.LocalLoad, 4, *outputs.Global, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.LocalLoad, 4, *outputs.Imported, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.LocalLoad, 4, *outputs.Alloca1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.LocalLoad, 4, *outputs.Alloca2, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.LocalLoad, 4, *outputs.Alloca4, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.LocalLoad, 4, *outputs.P, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.LocalLoad, 4, *outputs.Q, 4, AliasAnalysis::NoAlias);

  // The pointer value loaded from imported is just like the one loaded from global
  Expect(aa, *outputs.ImportedLoad, 4, *outputs.Global, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ImportedLoad, 4, *outputs.Imported, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ImportedLoad, 4, *outputs.Alloca1, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ImportedLoad, 4, *outputs.Local, 4, AliasAnalysis::NoAlias);

  // The pointer we get from getPtr() can point to anything that is externally available
  Expect(aa, *outputs.R, 4, *outputs.Global, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.R, 4, *outputs.Imported, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.R, 4, *outputs.Alloca1, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.R, 4, *outputs.Alloca2, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.R, 4, *outputs.Alloca4, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.R, 4, *outputs.P, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.R, 4, *outputs.Q, 4, AliasAnalysis::MayAlias);

  // It can not point to alloca3, as it never escaped the module
  Expect(aa, *outputs.R, 4, *outputs.Alloca3, 4, AliasAnalysis::NoAlias);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/PointsToGraphAliasAnalysisTests-TestPtGAliasAnalysis",
    TestPtGAliasAnalysis);

/**
 * This class sets up an RVSDG representing the following code:
 *
 * \code{.c}
 *   extern int32 globalInt;
 *   extern int64 globalLong;
 *
 *   void func(int32* p, int offset) {
 *
 *     int32* intWithOffset = (int32*) ((char*)globalInt + offset);
 *     int64* longWithOffset = (int64*) ((char*)globallong + offset);
 *
 *     // All alias queries happen here
 *   }
 * \endcode
 */
class PtGAliasAnalysisTestOffsets final : public jlm::tests::RvsdgTest
{
  struct Outputs
  {
    jlm::rvsdg::Output * GlobalInt = {};
    jlm::rvsdg::Output * GlobalLong = {};
    jlm::rvsdg::Output * Func = {};
    jlm::rvsdg::Output * P = {};
    jlm::rvsdg::Output * Offset = {};
    jlm::rvsdg::Output * IntWithOffset = {};
    jlm::rvsdg::Output * LongWithOffset = {};
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
    const auto byteType = rvsdg::BitType::Create(8);
    const auto int32Type = rvsdg::BitType::Create(32);
    const auto int64Type = rvsdg::BitType::Create(64);
    const auto ioStateType = IOStateType::Create();
    const auto memoryStateType = MemoryStateType::Create();

    const auto funcType = rvsdg::FunctionType::Create(
        { pointerType, int32Type, ioStateType, memoryStateType },
        { ioStateType, memoryStateType });

    Outputs_.GlobalInt =
        &GraphImport::Create(rvsdg, int32Type, pointerType, "globalInt", Linkage::externalLinkage);

    Outputs_.GlobalLong =
        &GraphImport::Create(rvsdg, int64Type, pointerType, "globalLong", Linkage::externalLinkage);

    // Setup the function "func"
    {
      auto & lambdaNode = *rvsdg::LambdaNode::Create(
          rvsdg.GetRootRegion(),
          LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

      Outputs_.P = lambdaNode.GetFunctionArguments()[0];
      Outputs_.Offset = lambdaNode.GetFunctionArguments()[1];
      auto ioState = lambdaNode.GetFunctionArguments()[2];
      auto memoryState = lambdaNode.GetFunctionArguments()[3];

      const auto globalIntCtxVar = lambdaNode.AddContextVar(*Outputs_.GlobalInt).inner;
      const auto globalLongCtxVar = lambdaNode.AddContextVar(*Outputs_.GlobalLong).inner;

      Outputs_.IntWithOffset = GetElementPtrOperation::Create(
          globalIntCtxVar,
          { Outputs_.Offset },
          byteType,
          pointerType);
      Outputs_.LongWithOffset = GetElementPtrOperation::Create(
          globalLongCtxVar,
          { Outputs_.Offset },
          byteType,
          pointerType);

      lambdaNode.finalize({ ioState, memoryState });
      Outputs_.Func = lambdaNode.output();
    }
    // Ensure func is being called from external modules
    rvsdg::GraphExport::Create(*Outputs_.Func, "func");

    return rvsdgModule;
  }

  Outputs Outputs_ = {};
};

void
TestPtGAliasAnalysisOffsets()
{
  using namespace jlm::llvm::aa;

  // Arrange
  PtGAliasAnalysisTestOffsets rvsdg;
  rvsdg.InitializeTest();
  const auto & outputs = rvsdg.GetOutputs();

  // jlm::rvsdg::view(&rvsdg.graph().GetRootRegion(), stdout);

  Andersen andersen;
  auto pointsToGraph = andersen.Analyze(rvsdg.module());
  // std::cout << PointsToGraph::ToDot(*pointsToGraph) << std::endl;
  PointsToGraphAliasAnalysis aa(*pointsToGraph);

  // Assert

  // Distinct global variables do not alias
  Expect(aa, *outputs.GlobalInt, 4, *outputs.GlobalLong, 4, AliasAnalysis::NoAlias);

  // The pointer argument can alias with either
  Expect(aa, *outputs.P, 4, *outputs.GlobalInt, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.P, 8, *outputs.GlobalLong, 8, AliasAnalysis::MayAlias);

  // If the accessed size is too large, the pointer argument can't access smaller memory objects
  Expect(aa, *outputs.P, 5, *outputs.GlobalInt, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.P, 9, *outputs.GlobalLong, 8, AliasAnalysis::NoAlias);

  // With small access sizes, the unknown offset causes MayAlias
  Expect(aa, *outputs.IntWithOffset, 2, *outputs.GlobalInt, 2, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.LongWithOffset, 2, *outputs.GlobalLong, 2, AliasAnalysis::MayAlias);

  // Even with an unknown offset, the pointers are not mixed
  Expect(aa, *outputs.IntWithOffset, 2, *outputs.GlobalLong, 2, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.LongWithOffset, 2, *outputs.GlobalInt, 2, AliasAnalysis::NoAlias);

  // With full access sizes, the unknown offset can be ignored, and we get MustAlias
  Expect(aa, *outputs.IntWithOffset, 4, *outputs.GlobalInt, 4, AliasAnalysis::MustAlias);
  Expect(aa, *outputs.LongWithOffset, 8, *outputs.GlobalLong, 8, AliasAnalysis::MustAlias);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/PointsToGraphAliasAnalysisTests-TestPtGAliasAnalysisOffsets",
    TestPtGAliasAnalysisOffsets);
