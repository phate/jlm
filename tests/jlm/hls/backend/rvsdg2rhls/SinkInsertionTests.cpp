/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rvsdg2rhls/add-sinks.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(SinkInsertionTests, SinkInsertion)
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = TestType::createValueType();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto argument = lambdaNode->GetFunctionArguments()[0];

  auto structuralNode = TestStructuralNode::create(lambdaNode->subregion(), 1);
  const auto inputVar0 = structuralNode->addInputWithArguments(*argument);
  const auto inputVar1 = structuralNode->addInputWithArguments(*argument);

  const auto outputVar0 = structuralNode->addOutputWithResults({ inputVar1.argument[0] });
  const auto outputVar1 = structuralNode->addOutputWithResults({ inputVar1.argument[0] });

  auto lambdaOutput = lambdaNode->finalize({ outputVar1.output });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  SinkInsertion::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  EXPECT_EQ(structuralNode->subregion(0)->numNodes(), 1);
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 2);

  // The sink insertion pass should have inserted a SinkOperation node at output o0
  {
    EXPECT_EQ(outputVar0.output->nusers(), 1);
    EXPECT_TRUE(IsOwnerNodeOperation<SinkOperation>(*outputVar0.output->Users().begin()));
  }

  // The sink insertion pass should have inserted a SinkOperation node at the argument of i0
  {
    auto & i0Argument = *inputVar0.argument[0];
    EXPECT_EQ(i0Argument.nusers(), 1);
    EXPECT_TRUE(IsOwnerNodeOperation<SinkOperation>(*i0Argument.Users().begin()));
  }
}
