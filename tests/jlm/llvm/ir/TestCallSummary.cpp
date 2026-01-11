/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(CallSummaryTests, TestCallSummaryComputationDead)
{
  using namespace jlm;

  // Arrange
  auto vt = rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { vt });

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          functionType,
          "f",
          jlm::llvm::Linkage::externalLinkage));

  auto result = rvsdg::TestOperation::createNode(lambdaNode->subregion(), {}, { vt })->output(0);

  lambdaNode->finalize({ result });

  // Act
  auto callSummary = jlm::llvm::ComputeCallSummary(*lambdaNode);

  // Assert
  EXPECT_TRUE(callSummary.IsDead());

  EXPECT_FALSE(callSummary.IsExported());
  EXPECT_FALSE(callSummary.IsOnlyExported());
  EXPECT_EQ(callSummary.GetRvsdgExport(), nullptr);
  EXPECT_FALSE(callSummary.HasOnlyDirectCalls());
}

TEST(CallSummaryTests, TestCallSummaryComputationExport)
{
  using namespace jlm;

  // Arrange
  auto vt = rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { vt });

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          functionType,
          "f",
          jlm::llvm::Linkage::externalLinkage));

  auto result = rvsdg::TestOperation::createNode(lambdaNode->subregion(), {}, { vt })->output(0);

  auto lambdaOutput = lambdaNode->finalize({ result });
  auto & rvsdgExport = rvsdg::GraphExport::Create(*lambdaOutput, "f");

  // Act
  auto callSummary = jlm::llvm::ComputeCallSummary(*lambdaNode);

  // Assert
  EXPECT_TRUE(callSummary.IsExported());
  EXPECT_TRUE(callSummary.IsOnlyExported());
  EXPECT_EQ(callSummary.GetRvsdgExport(), &rvsdgExport);

  EXPECT_FALSE(callSummary.IsDead());
  EXPECT_FALSE(callSummary.HasOnlyDirectCalls());
}

TEST(CallSummaryTests, TestCallSummaryComputationDirectCalls)
{
  using namespace jlm;

  // Arrange
  auto vt = rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::IOStateType::Create(), jlm::llvm::MemoryStateType::Create() },
      { vt, jlm::llvm::IOStateType::Create(), jlm::llvm::MemoryStateType::Create() });

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto SetupLambdaX = [&]()
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        jlm::llvm::LlvmLambdaOperation::Create(
            functionType,
            "x",
            jlm::llvm::Linkage::externalLinkage));
    auto iOStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];

    auto result = rvsdg::TestOperation::createNode(lambdaNode->subregion(), {}, { vt })->output(0);

    return lambdaNode->finalize({ result, iOStateArgument, memoryStateArgument });
  };

  auto SetupLambdaY = [&](rvsdg::Output & lambdaX)
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        jlm::llvm::LlvmLambdaOperation::Create(
            functionType,
            "y",
            jlm::llvm::Linkage::externalLinkage));
    auto iOStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto lambdaXCv = lambdaNode->AddContextVar(lambdaX).inner;

    auto callResults = jlm::llvm::CallOperation::Create(
        lambdaXCv,
        functionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambdaNode->finalize(callResults);
    rvsdg::GraphExport::Create(*lambdaOutput, "y");

    return lambdaOutput;
  };

  auto SetupLambdaZ = [&](rvsdg::Output & lambdaX, rvsdg::Output & lambdaY)
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        jlm::llvm::LlvmLambdaOperation::Create(
            functionType,
            "y",
            jlm::llvm::Linkage::externalLinkage));
    auto iOStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto lambdaXCv = lambdaNode->AddContextVar(lambdaX).inner;
    auto lambdaYCv = lambdaNode->AddContextVar(lambdaY).inner;

    auto callXResults = jlm::llvm::CallOperation::Create(
        lambdaXCv,
        functionType,
        { iOStateArgument, memoryStateArgument });
    auto callYResults = jlm::llvm::CallOperation::Create(
        lambdaYCv,
        functionType,
        { callXResults[1], callXResults[2] });

    auto result = rvsdg::TestOperation::createNode(
                      lambdaNode->subregion(),
                      { callXResults[0], callYResults[0] },
                      { vt })
                      ->output(0);

    auto lambdaOutput = lambdaNode->finalize({ result, callYResults[1], callYResults[2] });
    rvsdg::GraphExport::Create(*lambdaOutput, "z");

    return lambdaOutput;
  };

  auto lambdaX = SetupLambdaX();
  auto lambdaY = SetupLambdaY(*lambdaX);
  auto lambdaZ = SetupLambdaZ(*lambdaX, *lambdaY);

  // Act
  auto lambdaXCallSummary =
      jlm::llvm::ComputeCallSummary(rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaX));
  auto lambdaYCallSummary =
      jlm::llvm::ComputeCallSummary(rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaY));
  auto lambdaZCallSummary =
      jlm::llvm::ComputeCallSummary(rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaZ));

  // Assert
  EXPECT_TRUE(lambdaXCallSummary.HasOnlyDirectCalls());
  EXPECT_EQ(lambdaXCallSummary.NumDirectCalls(), 2u);
  EXPECT_FALSE(lambdaXCallSummary.IsDead());
  EXPECT_FALSE(lambdaXCallSummary.IsExported());
  EXPECT_FALSE(lambdaXCallSummary.IsOnlyExported());

  EXPECT_FALSE(lambdaYCallSummary.IsDead());
  EXPECT_FALSE(lambdaYCallSummary.HasOnlyDirectCalls());
  EXPECT_EQ(lambdaYCallSummary.NumDirectCalls(), 1u);
  EXPECT_TRUE(lambdaYCallSummary.IsExported());
  EXPECT_FALSE(lambdaYCallSummary.IsOnlyExported());

  EXPECT_FALSE(lambdaZCallSummary.IsDead());
  EXPECT_FALSE(lambdaZCallSummary.HasOnlyDirectCalls());
  EXPECT_EQ(lambdaZCallSummary.NumDirectCalls(), 0u);
  EXPECT_TRUE(lambdaZCallSummary.IsExported());
  EXPECT_TRUE(lambdaZCallSummary.IsOnlyExported());
}

TEST(CallSummaryTests, TestCallSummaryComputationIndirectCalls)
{
  using namespace jlm::llvm;

  // Arrange
  IndirectCallTest1 test;
  test.module();

  // Act
  auto lambdaThreeCallSummary = jlm::llvm::ComputeCallSummary(test.GetLambdaThree());
  auto lambdaFourCallSummary = jlm::llvm::ComputeCallSummary(test.GetLambdaFour());
  auto lambdaIndcallCallSummary = jlm::llvm::ComputeCallSummary(test.GetLambdaIndcall());
  auto lambdaTestCallSummary = jlm::llvm::ComputeCallSummary(test.GetLambdaTest());

  // Assert
  EXPECT_FALSE(lambdaThreeCallSummary.HasOnlyDirectCalls());
  EXPECT_EQ(lambdaThreeCallSummary.NumDirectCalls(), 0u);
  EXPECT_FALSE(lambdaThreeCallSummary.IsDead());
  EXPECT_FALSE(lambdaThreeCallSummary.IsExported());
  EXPECT_FALSE(lambdaThreeCallSummary.IsOnlyExported());
  EXPECT_EQ(lambdaThreeCallSummary.NumOtherUsers(), 1u);

  EXPECT_FALSE(lambdaFourCallSummary.HasOnlyDirectCalls());
  EXPECT_EQ(lambdaFourCallSummary.NumDirectCalls(), 0u);
  EXPECT_FALSE(lambdaFourCallSummary.IsDead());
  EXPECT_FALSE(lambdaFourCallSummary.IsExported());
  EXPECT_FALSE(lambdaFourCallSummary.IsOnlyExported());
  EXPECT_EQ(lambdaFourCallSummary.NumOtherUsers(), 1u);

  EXPECT_TRUE(lambdaIndcallCallSummary.HasOnlyDirectCalls());
  EXPECT_EQ(lambdaIndcallCallSummary.NumDirectCalls(), 2u);
  EXPECT_FALSE(lambdaIndcallCallSummary.IsDead());
  EXPECT_FALSE(lambdaIndcallCallSummary.IsExported());
  EXPECT_FALSE(lambdaIndcallCallSummary.IsOnlyExported());
  EXPECT_EQ(lambdaIndcallCallSummary.NumOtherUsers(), 0u);

  EXPECT_FALSE(lambdaTestCallSummary.HasOnlyDirectCalls());
  EXPECT_EQ(lambdaTestCallSummary.NumDirectCalls(), 0u);
  EXPECT_FALSE(lambdaTestCallSummary.IsDead());
  EXPECT_TRUE(lambdaTestCallSummary.IsExported());
  EXPECT_TRUE(lambdaTestCallSummary.IsOnlyExported());
  EXPECT_EQ(lambdaTestCallSummary.NumOtherUsers(), 0u);
}

TEST(CallSummaryTests, TestCallSummaryComputationFunctionPointerInDelta)
{
  using namespace jlm::llvm;

  // Arrange
  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create({ valueType }, { valueType });

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg->GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  lambdaNode->finalize({ lambdaNode->GetFunctionArguments()[0] });

  auto deltaNode = jlm::rvsdg::DeltaNode::Create(
      &rvsdg->GetRootRegion(),
      jlm::llvm::DeltaOperation::Create(functionType, "fp", Linkage::externalLinkage, "", false));
  auto argument = deltaNode->AddContextVar(*lambdaNode->output()).inner;
  deltaNode->finalize(argument);

  jlm::rvsdg::GraphExport::Create(deltaNode->output(), "fp");

  // Act
  auto callSummary = jlm::llvm::ComputeCallSummary(*lambdaNode);

  // Assert
  EXPECT_EQ(callSummary.NumOtherUsers(), 1u);
  EXPECT_TRUE(callSummary.HasOnlyOtherUsages());
}

TEST(CallSummaryTests, TestCallSummaryComputationLambdaResult)
{
  using namespace jlm::llvm;

  // Arrange
  jlm::rvsdg::Graph rvsdg;

  auto pointerType = PointerType::Create();
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionTypeG = jlm::rvsdg::FunctionType::Create({ valueType }, { valueType });
  auto functionTypeF = jlm::rvsdg::FunctionType::Create({ valueType }, { PointerType::Create() });

  auto lambdaNodeG = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionTypeG, "g", Linkage::externalLinkage));
  auto lambdaOutputG = lambdaNodeG->finalize({ lambdaNodeG->GetFunctionArguments()[0] });

  auto lambdaNodeF = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionTypeF, "f", Linkage::externalLinkage));
  auto lambdaGArgument = lambdaNodeF->AddContextVar(*lambdaOutputG).inner;
  auto ptr =
      jlm::rvsdg::CreateOpNode<FunctionToPointerOperation>({ lambdaGArgument }, functionTypeG)
          .output(0);
  auto lambdaOutputF = lambdaNodeF->finalize({ ptr });

  jlm::rvsdg::GraphExport::Create(*lambdaOutputF, "f");

  // Act
  auto callSummary = jlm::llvm::ComputeCallSummary(*lambdaNodeG);

  // Assert
  EXPECT_EQ(callSummary.NumOtherUsers(), 1u);
  EXPECT_TRUE(callSummary.HasOnlyOtherUsages());
}
