/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/opt/IOStateElimination.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(IOStateEliminationTests, testCall)
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  auto ioStateType = IOStateType::Create();
  const auto functionType =
      FunctionType::Create({ ioStateType, memoryStateType }, { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & i0 = jlm::rvsdg::GraphImport::Create(rvsdg, functionType, "g");

  const auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto ioStateArgument = lambdaNode->GetFunctionArguments()[0];
  auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];
  const auto functionCv = lambdaNode->AddContextVar(i0);

  auto & callNode = CallOperation::CreateNode(
      functionCv.inner,
      functionType,
      { ioStateArgument, memoryStateArgument });

  const auto lambdaOutput = lambdaNode->finalize({ callNode.output(0), callNode.output(1) });

  GraphExport::Create(*lambdaOutput, "f");

  view(rvsdg, stdout);

  // Act
  IOStateElimination ioStateElimination;
  jlm::util::StatisticsCollector statisticsCollector;
  ioStateElimination.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  EXPECT_TRUE(callNode.output(0)->IsDead());
  EXPECT_EQ(lambdaNode->GetFunctionResults()[0]->origin(), ioStateArgument);
}

TEST(IOStateEliminationTests, testNesting)
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create(
      { controlType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  const auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];
  auto memoryStateArgument = lambdaNode->GetFunctionArguments()[2];

  auto gammaNode = GammaNode::create(controlArgument, 2);

  auto entryVar = gammaNode->AddEntryVar(ioStateArgument);

  auto node1 = TestOperation::createNode(
      gammaNode->subregion(0),
      { entryVar.branchArgument[0] },
      { ioStateType });

  auto node2 =
      TestOperation::createNode(gammaNode->subregion(0), { node1->output(0) }, { ioStateType });

  auto exitVar = gammaNode->AddExitVar({ node2->output(0), entryVar.branchArgument[1] });

  const auto lambdaOutput = lambdaNode->finalize({ exitVar.output, memoryStateArgument });

  GraphExport::Create(*lambdaOutput, "f");

  view(rvsdg, stdout);

  // Act
  IOStateElimination ioStateElimination;
  jlm::util::StatisticsCollector statisticsCollector;
  ioStateElimination.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  EXPECT_TRUE(node1->output(0)->IsDead());
  EXPECT_TRUE(node2->output(0)->IsDead());
  EXPECT_TRUE(exitVar.output->IsDead());
  EXPECT_EQ(lambdaNode->GetFunctionResults()[0]->origin(), ioStateArgument);

  EXPECT_EQ(gammaNode->GetEntryVars().size(), 3);
}
