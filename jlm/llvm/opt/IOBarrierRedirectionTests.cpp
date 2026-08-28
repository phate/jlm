/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/IOBarrierRedirection.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

static void
runIOBarrierRedirection(LlvmRvsdgModule & rvsdgModule)
{
  IOBarrierRedirection ioBarrierRedirection;
  util::StatisticsCollector statisticsCollector;
  ioBarrierRedirection.Run(rvsdgModule, statisticsCollector);
}

TEST(IOBarrierRedirectionTests, testLoadGammaDependence)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i32Type = BitType::Create(32);
  auto controlType = ControlType::Create(2);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto functionType = FunctionType::Create({ pointerType, controlType, ioStateType }, { i32Type });

  auto rvsdgModule = LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  auto ptrArgument = lambdaNode->GetFunctionArguments()[0];
  auto controlArgument = lambdaNode->GetFunctionArguments()[1];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];

  auto outerGammaNode = GammaNode::create(controlArgument, 2);
  auto outerPtrEntryVar = outerGammaNode->AddEntryVar(ptrArgument);
  auto outerIOStateEntryVar = outerGammaNode->AddEntryVar(ioStateArgument);

  // outerGammaNode - subregion 0
  auto & outerIOBarrierNode = IOBarrierOperation::createNode(
      *outerPtrEntryVar.branchArgument[0],
      *outerIOStateEntryVar.branchArgument[0]);
  auto & outerLoadNode =
      LoadNonVolatileOperation::CreateNode(*outerIOBarrierNode.output(0), {}, i32Type, 4);

  auto & matchNode = MatchOperation::CreateNode(
      LoadOperation::LoadedValueOutput(outerLoadNode),
      { { 1, 1 } },
      0,
      2);

  auto innerGammaNode = GammaNode::create(matchNode.output(0), 2);
  auto innerPtrEntryVar = innerGammaNode->AddEntryVar(outerPtrEntryVar.branchArgument[0]);
  auto innerIOStateEntryVar = innerGammaNode->AddEntryVar(outerIOStateEntryVar.branchArgument[0]);
  auto innerI32EntryVar =
      innerGammaNode->AddEntryVar(&LoadOperation::LoadedValueOutput(outerLoadNode));

  // innerGammaNode - subregion 0
  auto & innerIOBarrierNode = IOBarrierOperation::createNode(
      *innerPtrEntryVar.branchArgument[0],
      *innerIOStateEntryVar.branchArgument[0]);
  auto & innerLoadNode =
      LoadNonVolatileOperation::CreateNode(*innerIOBarrierNode.output(0), {}, i32Type, 4);

  // innerGammaNode - subregion 1
  // Nothing to be done

  // innerGammaNode - finalize
  auto innerI32ExitVar = innerGammaNode->AddExitVar(
      { &LoadOperation::LoadedValueOutput(innerLoadNode), innerI32EntryVar.branchArgument[1] });

  // outerGammaNode - subregion 1
  auto testNode = TestOperation::createNode(outerGammaNode->subregion(1), {}, { i32Type });

  // outerGammaNode - finalize
  auto outerI32ExitVar =
      outerGammaNode->AddExitVar({ innerI32ExitVar.output, testNode->output(0) });

  auto lambdaOutput = lambdaNode->finalize({ outerI32ExitVar.output });
  GraphExport::Create(*lambdaOutput, "test");

  // Act
  runIOBarrierRedirection(*rvsdgModule);

  // Assert
  // We expect the IOBarrierOperation node in innerGamma subregion 0 to be eliminated
  EXPECT_FALSE(Region::containsOperation<IOBarrierOperation>(*innerGammaNode->subregion(0), true));

  // We expect the origin of the innerPtrEntryVar to originate now from the outerIOBarrierNode
  EXPECT_EQ(innerPtrEntryVar.input->origin(), outerIOBarrierNode.output(0));
}

TEST(IOBarrierRedirectionTests, testLoadGammaIndependence)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i32Type = BitType::Create(32);
  auto controlType = ControlType::Create(2);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto functionType = FunctionType::Create({ pointerType, controlType, ioStateType }, { i32Type });

  auto rvsdgModule = LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  auto ptrArgument = lambdaNode->GetFunctionArguments()[0];
  auto controlArgument = lambdaNode->GetFunctionArguments()[1];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];

  auto outerGammaNode = GammaNode::create(controlArgument, 2);
  auto outerPtrEntryVar = outerGammaNode->AddEntryVar(ptrArgument);
  auto outerIOStateEntryVar = outerGammaNode->AddEntryVar(ioStateArgument);

  // outerGammaNode - subregion 0
  auto predicateNode = TestOperation::createNode(outerGammaNode->subregion(0), {}, { controlType });

  auto innerGammaNode = GammaNode::create(predicateNode->output(0), 2);
  auto innerPtrEntryVar = innerGammaNode->AddEntryVar(outerPtrEntryVar.branchArgument[0]);
  auto innerIOStateEntryVar = innerGammaNode->AddEntryVar(outerIOStateEntryVar.branchArgument[0]);

  // innerGammaNode - subregion 0
  auto & innerIOBarrierNode = IOBarrierOperation::createNode(
      *innerPtrEntryVar.branchArgument[0],
      *innerIOStateEntryVar.branchArgument[0]);
  auto & innerLoadNode =
      LoadNonVolatileOperation::CreateNode(*innerIOBarrierNode.output(0), {}, i32Type, 4);

  // innerGammaNode - subregion 1
  auto innerI32ConstantNode =
      TestOperation::createNode(innerGammaNode->subregion(1), {}, { i32Type });

  // innerGammaNode - finalize
  auto innerI32ExitVar = innerGammaNode->AddExitVar(
      { &LoadOperation::LoadedValueOutput(innerLoadNode), innerI32ConstantNode->output(0) });

  auto & outerIOBarrierNode = IOBarrierOperation::createNode(
      *outerPtrEntryVar.branchArgument[0],
      *outerIOStateEntryVar.branchArgument[0]);
  auto & outerLoadNode =
      LoadNonVolatileOperation::CreateNode(*outerIOBarrierNode.output(0), {}, i32Type, 4);

  auto testNode = TestOperation::createNode(
      outerGammaNode->subregion(0),
      { innerI32ExitVar.output, &LoadOperation::LoadedValueOutput(outerLoadNode) },
      { i32Type });

  // outerGammaNode - subregion 1
  auto outerI32ConstantNode =
      TestOperation::createNode(outerGammaNode->subregion(1), {}, { i32Type });

  // outerGammaNode - finalize
  auto outerI32ExitVar =
      outerGammaNode->AddExitVar({ testNode->output(0), outerI32ConstantNode->output(0) });

  auto lambdaOutput = lambdaNode->finalize({ outerI32ExitVar.output });
  GraphExport::Create(*lambdaOutput, "test");

  // Act
  runIOBarrierRedirection(*rvsdgModule);

  // Assert
  // We expect the IOBarrierOperation node in innerGamma subregion 0 to be eliminated
  EXPECT_FALSE(Region::containsOperation<IOBarrierOperation>(*innerGammaNode->subregion(0), true));

  // We expect the origin of the innerPtrEntryVar to originate now from the outerIOBarrierNode
  EXPECT_EQ(innerPtrEntryVar.input->origin(), outerIOBarrierNode.output(0));
}

}
