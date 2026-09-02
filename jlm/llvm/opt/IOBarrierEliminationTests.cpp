/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/IOBarrierElimination.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

static void
runIOBarrierElimination(LlvmRvsdgModule & rvsdgModule)
{

  util::StatisticsCollector statisticsCollector;
  IOBarrierElimination ioBarrierElimination;
  ioBarrierElimination.Run(rvsdgModule, statisticsCollector);
}

TEST(IOBarrierEliminationTests, testLambdaArgument)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i32Type = BitType::Create(32);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto functionType = FunctionType::Create({ pointerType, ioStateType }, { i32Type, ioStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  auto ptrArgument = lambdaNode->GetFunctionArguments()[0];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];

  auto & ioBarrierNode = IOBarrierOperation::createNode(*ptrArgument, *ioStateArgument);

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*ioBarrierNode.output(0), {}, i32Type, 4);

  auto lambdaOutput = lambdaNode->finalize({ loadNode.output(0), ioStateArgument });
  GraphExport::Create(*lambdaOutput, "test");

  // Act
  runIOBarrierElimination(*rvsdgModule);

  // Assert
  // We expect the IOBarrier node to be eliminated
  EXPECT_FALSE(Region::containsOperation<IOBarrierOperation>(rvsdg.GetRootRegion(), true));
}

TEST(IOBarrierEliminationTests, testSizeIsRespected)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i8Type = BitType::Create(8);
  auto i32Type = BitType::Create(32);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto functionType =
      FunctionType::Create({ pointerType, ioStateType }, { i8Type, i32Type, ioStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  auto ptrArgument = lambdaNode->GetFunctionArguments()[0];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];

  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(*ptrArgument, {}, i8Type, 4);

  auto testNode =
      TestOperation::createNode(lambdaNode->subregion(), { ioStateArgument }, { ioStateType });

  auto & ioBarrierNode = IOBarrierOperation::createNode(*ptrArgument, *testNode->output(0));

  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(*ioBarrierNode.output(0), {}, i32Type, 4);

  auto lambdaOutput =
      lambdaNode->finalize({ loadNode1.output(0), loadNode2.output(0), testNode->output(0) });
  GraphExport::Create(*lambdaOutput, "test");

  // Act
  runIOBarrierElimination(*rvsdgModule);

  // Assert
  // We expect the IOBarrier node to NOT be eliminated as loadNode1 marks the pointer argument only
  // dereferenceable with size i8, but loadNode2 requires size i32.
  EXPECT_TRUE(Region::containsOperation<IOBarrierOperation>(rvsdg.GetRootRegion(), true));
}

TEST(IOBarrierEliminationTests, testSuccess)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i32Type = BitType::Create(32);
  auto i64Type = BitType::Create(64);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto functionType =
      FunctionType::Create({ pointerType, ioStateType }, { i64Type, i32Type, ioStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  auto ptrArgument = lambdaNode->GetFunctionArguments()[0];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];

  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(*ptrArgument, {}, i64Type, 4);

  auto testNode =
      TestOperation::createNode(lambdaNode->subregion(), { ioStateArgument }, { ioStateType });

  auto & ioBarrierNode = IOBarrierOperation::createNode(*ptrArgument, *testNode->output(0));

  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(*ioBarrierNode.output(0), {}, i32Type, 4);

  auto lambdaOutput =
      lambdaNode->finalize({ loadNode1.output(0), loadNode2.output(0), testNode->output(0) });
  GraphExport::Create(*lambdaOutput, "test");

  // Act
  runIOBarrierElimination(*rvsdgModule);

  // Assert
  // We expect the IOBarrier node to be eliminated as loadNode1 marks the pointer argument
  // dereferenceable with size i64, but loadNode2 only requires size i32.
  EXPECT_FALSE(Region::containsOperation<IOBarrierOperation>(rvsdg.GetRootRegion(), true));
}

TEST(IOBarrierEliminationTests, testInvidiualIOBarrierUserRerouting)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i8Type = BitType::Create(8);
  auto i32Type = BitType::Create(32);
  auto i64Type = BitType::Create(64);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto functionType =
      FunctionType::Create({ pointerType, ioStateType }, { i32Type, i8Type, i64Type, ioStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  auto ptrArgument = lambdaNode->GetFunctionArguments()[0];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];

  auto & load32Node = LoadNonVolatileOperation::CreateNode(*ptrArgument, {}, i32Type, 4);

  auto testNode =
      TestOperation::createNode(lambdaNode->subregion(), { ioStateArgument }, { ioStateType });

  auto & ioBarrierNode = IOBarrierOperation::createNode(*ptrArgument, *testNode->output(0));

  auto & load8Node = LoadNonVolatileOperation::CreateNode(*ioBarrierNode.output(0), {}, i8Type, 4);

  auto & load64Node =
      LoadNonVolatileOperation::CreateNode(*ioBarrierNode.output(0), {}, i64Type, 4);

  auto lambdaOutput = lambdaNode->finalize(
      { load32Node.output(0), load8Node.output(0), load64Node.output(0), testNode->output(0) });
  GraphExport::Create(*lambdaOutput, "test");

  // Act
  runIOBarrierElimination(*rvsdgModule);

  // Assert
  EXPECT_TRUE(Region::containsOperation<IOBarrierOperation>(rvsdg.GetRootRegion(), true));

  // We expect that the load8Node is not any longer barred behind the IOBarrier node as ptrArgument
  // is dereferenceable for 32 bits.
  EXPECT_EQ(LoadOperation::AddressInput(load8Node).origin(), ptrArgument);

  // We expect that the load64Node is still barred behind the IOBarrier node as ptrArgument is only
  // dereferenceable for 64 bits.
  EXPECT_EQ(LoadOperation::AddressInput(load64Node).origin(), ioBarrierNode.output(0));
}

TEST(IOBarrierEliminationTests, testGamma)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i32Type = BitType::Create(32);
  auto i64Type = BitType::Create(64);
  auto controlType = ControlType::Create(2);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto functionType = FunctionType::Create(
      { pointerType, controlType, ioStateType },
      { i32Type, i32Type, ioStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  auto ptrArgument = lambdaNode->GetFunctionArguments()[0];
  auto controlArgument = lambdaNode->GetFunctionArguments()[1];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*ptrArgument, {}, i32Type, 4);

  auto gammaNode = GammaNode::create(controlArgument, 2);
  auto ptrEntryVar = gammaNode->AddEntryVar(ptrArgument);
  auto ioStateEntryVar = gammaNode->AddEntryVar(ioStateArgument);

  // subregion 0
  auto & ioBarrierNode0 = IOBarrierOperation::createNode(
      *ptrEntryVar.branchArgument[0],
      *ioStateEntryVar.branchArgument[0]);
  auto & load32Node =
      LoadNonVolatileOperation::CreateNode(*ioBarrierNode0.output(0), {}, i32Type, 4);

  // subregion 1
  auto & ioBarrierNode1 = IOBarrierOperation::createNode(
      *ptrEntryVar.branchArgument[1],
      *ioStateEntryVar.branchArgument[1]);
  auto & load64Node =
      LoadNonVolatileOperation::CreateNode(*ioBarrierNode1.output(0), {}, i64Type, 4);
  auto testNode =
      TestOperation::createNode(gammaNode->subregion(1), { load64Node.output(0) }, { i32Type });

  // gamma exit
  auto i32ExitVar = gammaNode->AddExitVar({ load32Node.output(0), testNode->output(0) });
  auto ioStateExitVar = gammaNode->AddExitVar(
      { ioStateEntryVar.branchArgument[0], ioStateEntryVar.branchArgument[1] });

  auto lambdaOutput =
      lambdaNode->finalize({ loadNode.output(0), i32ExitVar.output, ioStateExitVar.output });
  GraphExport::Create(*lambdaOutput, "test");

  // Act
  runIOBarrierElimination(*rvsdgModule);

  // Assert
  // We expect the IOBarrierOperation node in gamma subregion 0 to be eliminated
  EXPECT_FALSE(Region::containsOperation<IOBarrierOperation>(*gammaNode->subregion(0), true));

  // We expect the IOBarrierOperation nodes in gamma subregion 1 NOT to be eliminated
  EXPECT_TRUE(Region::containsOperation<IOBarrierOperation>(*gammaNode->subregion(1), true));
}

TEST(IOBarrierEliminationTest, testNormalizeation)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto functionType = FunctionType::Create(
      { pointerType, ioStateType },
      { pointerType, pointerType, pointerType, pointerType, ioStateType });

  Graph rvsdg;

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  auto ptrArgument = lambdaNode->GetFunctionArguments()[0];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];

  auto & ioBarrierNode0 = IOBarrierOperation::createNode(*ptrArgument, *ioStateArgument);

  auto structuralNode = TestStructuralNode::create(lambdaNode->subregion(), 2);
  auto ptrInputVar = structuralNode->addInputWithArguments(*ptrArgument);
  auto ioStateInputVar = structuralNode->addInputWithArguments(*ioStateArgument);

  // subregion 0
  auto & ioBarrierNode1 =
      IOBarrierOperation::createNode(*ptrInputVar.argument[0], *ioStateInputVar.argument[0]);

  // subregion 1
  // Nothing needs to be done

  // finalize
  auto ptrOutputVar1 =
      structuralNode->addOutputWithResults({ ioBarrierNode1.output(0), ptrInputVar.argument[1] });
  auto ptrOutputVar2 =
      structuralNode->addOutputWithResults({ ptrInputVar.argument[0], ptrInputVar.argument[1] });
  auto ioStateOutputVar = structuralNode->addOutputWithResults(
      { ioStateInputVar.argument[0], ioStateInputVar.argument[1] });

  auto & ioBarrierNode2 =
      IOBarrierOperation::createNode(*ptrOutputVar1.output, *ioStateOutputVar.output);

  auto lambdaOutput = lambdaNode->finalize(
      { ioBarrierNode0.output(0),
        ioBarrierNode2.output(0),
        ptrOutputVar1.output,
        ptrOutputVar2.output,
        ioStateOutputVar.output });
  GraphExport::Create(*lambdaOutput, "test");

  // Act
  IOBarrierElimination::normalizeIOBarriers(rvsdg.GetRootRegion());

  // Assert
  EXPECT_EQ(ptrArgument->nusers(), 1);
  EXPECT_EQ(ptrInputVar.argument[0]->nusers(), 1);
  EXPECT_EQ(ptrOutputVar1.output->nusers(), 1);

  EXPECT_EQ(ptrInputVar.input->origin(), ioBarrierNode0.output(0));
  EXPECT_EQ(ptrOutputVar2.result[0]->origin(), ioBarrierNode1.output(0));
  EXPECT_EQ(lambdaNode->GetFunctionResults()[2]->origin(), ioBarrierNode2.output(0));
}

}
