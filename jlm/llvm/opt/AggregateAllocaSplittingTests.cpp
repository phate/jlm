/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/AggregateAllocaSplitting.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>

static void
assertAllocaWithType(const jlm::rvsdg::Output & output, const jlm::rvsdg::Type & type)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto [allocaNode, allocaOperation] =
      jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(output);
  EXPECT_TRUE(allocaNode && allocaOperation);
  EXPECT_EQ(*allocaOperation->allocatedType(), type);
}

TEST(AggregateAllocaSplittingTests, getElementPtrTest)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto bit32Type = BitType::Create(32);
  auto bit64Type = BitType::Create(64);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto structType = StructType::CreateIdentified({ bit32Type, bit64Type }, false);
  const auto functionType = FunctionType::Create({}, { memoryStateType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & zero32Node = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 0);
  auto & zero64Node = IntegerConstantOperation::Create(*lambdaNode->subregion(), 64, 0);
  auto & one32Node = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 1);
  auto & allocaNode = AllocaOperation::createNode(structType, *one32Node.output(0), 4);

  auto & gepXNode = GetElementPtrOperation::createNode(
      *allocaNode.output(0),
      { zero32Node.output(0), zero32Node.output(0) },
      bit32Type);
  auto & storeGepXNode = StoreNonVolatileOperation::CreateNode(
      *gepXNode.output(0),
      *zero32Node.output(0),
      { &AllocaOperation::getMemoryStateOutput(allocaNode) },
      4);

  auto & gepYNode = GetElementPtrOperation::createNode(
      *allocaNode.output(0),
      { zero32Node.output(0), one32Node.output(0) },
      bit64Type);
  auto & storeGepYNode = StoreNonVolatileOperation::CreateNode(
      *gepYNode.output(0),
      *zero64Node.output(0),
      { storeGepXNode.output(0) },
      4);

  auto lambdaOutput = lambdaNode->finalize({ storeGepYNode.output(0) });
  GraphExport::Create(*lambdaOutput, "f");

  // Act
  StatisticsCollector statisticsCollector;
  AggregateAllocaSplitting aggregateAllocaSplitting;
  aggregateAllocaSplitting.Run(rvsdgModule, statisticsCollector);

  // Assert
  // We expect two AllocaOperation, a MemoryStateMergeOperation, and a IntegerConstantOperation
  // node
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 8u);

  assertAllocaWithType(*StoreOperation::AddressInput(storeGepXNode).origin(), *bit32Type);
  assertAllocaWithType(*StoreOperation::AddressInput(storeGepYNode).origin(), *bit64Type);

  // Check memstate
  {
    auto [memoryMergeNode, memoryMergeOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(
            *StoreOperation::getMemoryStateInputs(storeGepXNode).begin()->origin());
    EXPECT_TRUE(memoryMergeNode && memoryMergeOperation);

    EXPECT_EQ(memoryMergeNode->ninputs(), 2u);
    assertAllocaWithType(*memoryMergeNode->input(0)->origin(), *bit32Type);
    assertAllocaWithType(*memoryMergeNode->input(1)->origin(), *bit64Type);
  }
}

TEST(AggregateAllocaSplittingTests, gammaTest)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto bit16Type = BitType::Create(16);
  auto bit32Type = BitType::Create(32);
  auto bit64Type = BitType::Create(64);
  const auto controlType = ControlType::Create(2);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto structType = StructType::CreateIdentified({ bit16Type, bit32Type, bit64Type }, false);
  const auto functionType = FunctionType::Create({ controlType }, { memoryStateType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];

  auto & zero = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 0);
  auto & one = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 1);
  auto & two = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 2);
  auto & allocaNode = AllocaOperation::createNode(structType, *one.output(0), 4);

  auto gammaNode = GammaNode::create(controlArgument, 2);
  auto baseAddressEntryVar = gammaNode->AddEntryVar(&AllocaOperation::getPointerOutput(allocaNode));
  auto memoryStateEntryVar =
      gammaNode->AddEntryVar(&AllocaOperation::getMemoryStateOutput(allocaNode));

  Node *storeGep0Node = nullptr, *storeGep1Node = nullptr;
  // Subregion 0
  {
    auto & zero16Node = IntegerConstantOperation::Create(*gammaNode->subregion(0), 16, 0);
    auto & zero32Node = IntegerConstantOperation::Create(*gammaNode->subregion(0), 32, 0);

    auto & gep0Node = GetElementPtrOperation::createNode(
        *baseAddressEntryVar.branchArgument[0],
        { zero32Node.output(0), zero32Node.output(0) },
        bit16Type);
    storeGep0Node = &StoreNonVolatileOperation::CreateNode(
        *gep0Node.output(0),
        *zero16Node.output(0),
        { memoryStateEntryVar.branchArgument[0] },
        4);
  }

  // Subregion 1
  {
    auto & zero32Node = IntegerConstantOperation::Create(*gammaNode->subregion(1), 32, 0);
    auto & one32Node = IntegerConstantOperation::Create(*gammaNode->subregion(1), 32, 1);

    auto & gep1Node = GetElementPtrOperation::createNode(
        *baseAddressEntryVar.branchArgument[1],
        { zero32Node.output(0), one32Node.output(0) },
        bit32Type);
    storeGep1Node = &StoreNonVolatileOperation::CreateNode(
        *gep1Node.output(0),
        *zero32Node.output(0),
        { memoryStateEntryVar.branchArgument[1] },
        4);
  }
  auto baseAddressExitVar = gammaNode->AddExitVar(baseAddressEntryVar.branchArgument);
  auto memoryStateExitVar =
      gammaNode->AddExitVar({ storeGep0Node->output(0), storeGep1Node->output(0) });

  auto & gep2Node = GetElementPtrOperation::createNode(
      *baseAddressExitVar.output,
      { zero.output(0), two.output(0) },
      bit64Type);

  auto & zero64Node = IntegerConstantOperation::Create(*lambdaNode->subregion(), 64, 0);
  auto & storeGep2Node = StoreNonVolatileOperation::CreateNode(
      *gep2Node.output(0),
      *zero64Node.output(0),
      { memoryStateExitVar.output },
      4);

  auto lambdaOutput = lambdaNode->finalize({ storeGep2Node.output(0) });
  GraphExport::Create(*lambdaOutput, "f");

  // Act
  StatisticsCollector statisticsCollector;
  AggregateAllocaSplitting aggregateAllocaSplitting;
  aggregateAllocaSplitting.Run(rvsdgModule, statisticsCollector);

  // Assert
  // Check gep0
  {
    auto & gammaInput =
        gammaNode->mapBranchArgumentToInput(*StoreOperation::AddressInput(*storeGep0Node).origin());
    assertAllocaWithType(*gammaInput.origin(), *bit16Type);
  }

  // Check gep1
  {
    auto & gammaInput =
        gammaNode->mapBranchArgumentToInput(*StoreOperation::AddressInput(*storeGep1Node).origin());
    assertAllocaWithType(*gammaInput.origin(), *bit32Type);
  }

  // Check gep2
  {
    assertAllocaWithType(*StoreOperation::AddressInput(storeGep2Node).origin(), *bit64Type);
  }

  // Check memState
  {
    auto [memoryMergeNode, memoryMergeOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(
            *memoryStateEntryVar.input->origin());
    EXPECT_TRUE(memoryMergeNode && memoryMergeOperation);

    EXPECT_EQ(memoryMergeNode->ninputs(), 3u);
    assertAllocaWithType(*memoryMergeNode->input(0)->origin(), *bit16Type);
    assertAllocaWithType(*memoryMergeNode->input(1)->origin(), *bit32Type);
    assertAllocaWithType(*memoryMergeNode->input(2)->origin(), *bit64Type);
  }
}

TEST(AggregateAllocaSplittingTests, thetaTest)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto bit16Type = BitType::Create(16);
  auto bit32Type = BitType::Create(32);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto structType = StructType::CreateIdentified({ bit16Type, bit32Type }, false);
  const auto functionType = FunctionType::Create({}, { memoryStateType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & one = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 1);
  auto & allocaNode = AllocaOperation::createNode(structType, *one.output(0), 4);

  Node * storeGep0Node = nullptr;
  auto thetaNode = ThetaNode::create(lambdaNode->subregion());
  auto addressLoopVar = thetaNode->AddLoopVar(&AllocaOperation::getPointerOutput(allocaNode));
  auto memoryStateLoopVar =
      thetaNode->AddLoopVar(&AllocaOperation::getMemoryStateOutput(allocaNode));
  {
    auto & zero16Node = IntegerConstantOperation::Create(*thetaNode->subregion(), 16, 0);
    auto & zero32Node = IntegerConstantOperation::Create(*thetaNode->subregion(), 32, 0);
    auto & gep0Node = GetElementPtrOperation::createNode(
        *addressLoopVar.pre,
        { zero32Node.output(0), zero32Node.output(0) },
        bit16Type);

    storeGep0Node = &StoreNonVolatileOperation::CreateNode(
        *gep0Node.output(0),
        *zero16Node.output(0),
        { memoryStateLoopVar.pre },
        4);

    memoryStateLoopVar.post->divert_to(storeGep0Node->output(0));
  }

  auto & zero32Node = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 0);
  auto & gep1Node = GetElementPtrOperation::createNode(
      *addressLoopVar.output,
      { zero32Node.output(0), one.output(0) },
      bit32Type);
  auto & storeGep1Node = StoreNonVolatileOperation::CreateNode(
      *gep1Node.output(0),
      *zero32Node.output(0),
      { memoryStateLoopVar.output },
      4);

  auto lambdaOutput = lambdaNode->finalize({ storeGep1Node.output(0) });
  GraphExport::Create(*lambdaOutput, "f");

  // Act
  StatisticsCollector statisticsCollector;
  AggregateAllocaSplitting aggregateAllocaSplitting;
  aggregateAllocaSplitting.Run(rvsdgModule, statisticsCollector);

  // Assert
  // Check gep0
  {
    const auto & tracedOutput = traceOutput(*StoreOperation::AddressInput(*storeGep0Node).origin());
    assertAllocaWithType(tracedOutput, *bit16Type);
  }

  // Check gep1
  {
    const auto & tracedOutput = traceOutput(*StoreOperation::AddressInput(storeGep1Node).origin());
    assertAllocaWithType(tracedOutput, *bit32Type);
  }

  // Check memstate
  {
    auto [memoryMergeNode, memoryMergeOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(
            *memoryStateLoopVar.input->origin());
    EXPECT_TRUE(memoryMergeNode && memoryMergeOperation);

    EXPECT_EQ(memoryMergeNode->ninputs(), 2u);
    assertAllocaWithType(*memoryMergeNode->input(0)->origin(), *bit16Type);
    assertAllocaWithType(*memoryMergeNode->input(1)->origin(), *bit32Type);
  }
}
