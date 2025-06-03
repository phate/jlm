/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

static void
TestCallSummaryComputationDead()
{
  using namespace jlm;

  // Arrange
  auto vt = tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { vt });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          functionType,
          "f",
          jlm::llvm::linkage::external_linkage));

  auto result = tests::create_testop(lambdaNode->subregion(), {}, { vt })[0];

  lambdaNode->finalize({ result });

  // Act
  auto callSummary = jlm::llvm::ComputeCallSummary(*lambdaNode);

  // Assert
  assert(callSummary.IsDead());

  assert(callSummary.IsExported() == false);
  assert(callSummary.IsOnlyExported() == false);
  assert(callSummary.GetRvsdgExport() == nullptr);
  assert(callSummary.HasOnlyDirectCalls() == false);
}

static void
TestCallSummaryComputationExport()
{
  using namespace jlm;

  // Arrange
  auto vt = tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { vt });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          functionType,
          "f",
          jlm::llvm::linkage::external_linkage));

  auto result = tests::create_testop(lambdaNode->subregion(), {}, { vt })[0];

  auto lambdaOutput = lambdaNode->finalize({ result });
  auto & rvsdgExport = jlm::llvm::GraphExport::Create(*lambdaOutput, "f");

  // Act
  auto callSummary = jlm::llvm::ComputeCallSummary(*lambdaNode);

  // Assert
  assert(callSummary.IsExported());
  assert(callSummary.IsOnlyExported());
  assert(callSummary.GetRvsdgExport() == &rvsdgExport);

  assert(callSummary.IsDead() == false);
  assert(callSummary.HasOnlyDirectCalls() == false);
}

static void
TestCallSummaryComputationDirectCalls()
{
  using namespace jlm;

  // Arrange
  auto vt = tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::IOStateType::Create(), jlm::llvm::MemoryStateType::Create() },
      { vt, jlm::llvm::IOStateType::Create(), jlm::llvm::MemoryStateType::Create() });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto SetupLambdaX = [&]()
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        jlm::llvm::LlvmLambdaOperation::Create(
            functionType,
            "x",
            jlm::llvm::linkage::external_linkage));
    auto iOStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];

    auto result = tests::create_testop(lambdaNode->subregion(), {}, { vt })[0];

    return lambdaNode->finalize({ result, iOStateArgument, memoryStateArgument });
  };

  auto SetupLambdaY = [&](rvsdg::Output & lambdaX)
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        jlm::llvm::LlvmLambdaOperation::Create(
            functionType,
            "y",
            jlm::llvm::linkage::external_linkage));
    auto iOStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto lambdaXCv = lambdaNode->AddContextVar(lambdaX).inner;

    auto callResults = jlm::llvm::CallOperation::Create(
        lambdaXCv,
        functionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambdaNode->finalize(callResults);
    jlm::llvm::GraphExport::Create(*lambdaOutput, "y");

    return lambdaOutput;
  };

  auto SetupLambdaZ = [&](rvsdg::Output & lambdaX, rvsdg::Output & lambdaY)
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        jlm::llvm::LlvmLambdaOperation::Create(
            functionType,
            "y",
            jlm::llvm::linkage::external_linkage));
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

    auto result = tests::create_testop(
        lambdaNode->subregion(),
        { callXResults[0], callYResults[0] },
        { vt })[0];

    auto lambdaOutput = lambdaNode->finalize({ result, callYResults[1], callYResults[2] });
    jlm::llvm::GraphExport::Create(*lambdaOutput, "z");

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
  assert(lambdaXCallSummary.HasOnlyDirectCalls());
  assert(lambdaXCallSummary.NumDirectCalls() == 2);
  assert(lambdaXCallSummary.IsDead() == false);
  assert(lambdaXCallSummary.IsExported() == false);
  assert(lambdaXCallSummary.IsOnlyExported() == false);

  assert(lambdaYCallSummary.IsDead() == false);
  assert(lambdaYCallSummary.HasOnlyDirectCalls() == false);
  assert(lambdaYCallSummary.NumDirectCalls() == 1);
  assert(lambdaYCallSummary.IsExported());
  assert(lambdaYCallSummary.IsOnlyExported() == false);

  assert(lambdaZCallSummary.IsDead() == false);
  assert(lambdaZCallSummary.HasOnlyDirectCalls() == false);
  assert(lambdaZCallSummary.NumDirectCalls() == 0);
  assert(lambdaZCallSummary.IsExported());
  assert(lambdaZCallSummary.IsOnlyExported());
}

static void
TestCallSummaryComputationIndirectCalls()
{
  // Arrange
  jlm::tests::IndirectCallTest1 test;
  test.module();

  // Act
  auto lambdaThreeCallSummary = jlm::llvm::ComputeCallSummary(test.GetLambdaThree());
  auto lambdaFourCallSummary = jlm::llvm::ComputeCallSummary(test.GetLambdaFour());
  auto lambdaIndcallCallSummary = jlm::llvm::ComputeCallSummary(test.GetLambdaIndcall());
  auto lambdaTestCallSummary = jlm::llvm::ComputeCallSummary(test.GetLambdaTest());

  // Assert
  assert(lambdaThreeCallSummary.HasOnlyDirectCalls() == false);
  assert(lambdaThreeCallSummary.NumDirectCalls() == 0);
  assert(lambdaThreeCallSummary.IsDead() == false);
  assert(lambdaThreeCallSummary.IsExported() == false);
  assert(lambdaThreeCallSummary.IsOnlyExported() == false);
  assert(lambdaThreeCallSummary.NumOtherUsers() == 1);

  assert(lambdaFourCallSummary.HasOnlyDirectCalls() == false);
  assert(lambdaFourCallSummary.NumDirectCalls() == 0);
  assert(lambdaFourCallSummary.IsDead() == false);
  assert(lambdaFourCallSummary.IsExported() == false);
  assert(lambdaFourCallSummary.IsOnlyExported() == false);
  assert(lambdaFourCallSummary.NumOtherUsers() == 1);

  assert(lambdaIndcallCallSummary.HasOnlyDirectCalls());
  assert(lambdaIndcallCallSummary.NumDirectCalls() == 2);
  assert(lambdaIndcallCallSummary.IsDead() == false);
  assert(lambdaIndcallCallSummary.IsExported() == false);
  assert(lambdaIndcallCallSummary.IsOnlyExported() == false);
  assert(lambdaIndcallCallSummary.NumOtherUsers() == 0);

  assert(lambdaTestCallSummary.HasOnlyDirectCalls() == false);
  assert(lambdaTestCallSummary.NumDirectCalls() == 0);
  assert(lambdaTestCallSummary.IsDead() == false);
  assert(lambdaTestCallSummary.IsExported());
  assert(lambdaTestCallSummary.IsOnlyExported());
  assert(lambdaTestCallSummary.NumOtherUsers() == 0);
}

static void
TestCallSummaryComputationFunctionPointerInDelta()
{
  using namespace jlm::llvm;

  // Arrange
  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create({ valueType }, { valueType });

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg->GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
  lambdaNode->finalize({ lambdaNode->GetFunctionArguments()[0] });

  auto deltaNode = delta::node::Create(
      &rvsdg->GetRootRegion(),
      functionType,
      "fp",
      linkage::external_linkage,
      "",
      false);
  auto argument = deltaNode->add_ctxvar(lambdaNode->output());
  deltaNode->finalize(argument);

  GraphExport::Create(*deltaNode->output(), "fp");

  // Act
  auto callSummary = jlm::llvm::ComputeCallSummary(*lambdaNode);

  // Assert
  assert(callSummary.NumOtherUsers() == 1);
  assert(callSummary.HasOnlyOtherUsages());
}

static void
TestCallSummaryComputationLambdaResult()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::rvsdg::Graph rvsdg;

  auto pointerType = PointerType::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto functionTypeG = jlm::rvsdg::FunctionType::Create({ valueType }, { valueType });
  auto functionTypeF = jlm::rvsdg::FunctionType::Create({ valueType }, { PointerType::Create() });

  auto lambdaNodeG = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionTypeG, "g", linkage::external_linkage));
  auto lambdaOutputG = lambdaNodeG->finalize({ lambdaNodeG->GetFunctionArguments()[0] });

  auto lambdaNodeF = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionTypeF, "f", linkage::external_linkage));
  auto lambdaGArgument = lambdaNodeF->AddContextVar(*lambdaOutputG).inner;
  auto ptr =
      jlm::rvsdg::CreateOpNode<FunctionToPointerOperation>({ lambdaGArgument }, functionTypeG)
          .output(0);
  auto lambdaOutputF = lambdaNodeF->finalize({ ptr });

  GraphExport::Create(*lambdaOutputF, "f");

  // Act
  auto callSummary = jlm::llvm::ComputeCallSummary(*lambdaNodeG);

  // Assert
  assert(callSummary.NumOtherUsers() == 1);
  assert(callSummary.HasOnlyOtherUsages());
}

static int
Test()
{
  TestCallSummaryComputationDead();
  TestCallSummaryComputationExport();
  TestCallSummaryComputationDirectCalls();
  TestCallSummaryComputationIndirectCalls();
  TestCallSummaryComputationFunctionPointerInDelta();
  TestCallSummaryComputationLambdaResult();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TestCallSummary", Test)
