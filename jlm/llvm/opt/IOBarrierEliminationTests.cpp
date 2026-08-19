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
#include <jlm/rvsdg/lambda.hpp>
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

}
