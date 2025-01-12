/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

static void
TestArgumentIterators()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  {
    auto functionType = FunctionType::Create({ vt }, { vt });

    auto lambda = lambda::node::create(
        &rvsdgModule.Rvsdg().GetRootRegion(),
        functionType,
        "f",
        linkage::external_linkage);
    lambda->finalize({ lambda->GetFunctionArguments()[0] });

    std::vector<const jlm::rvsdg::output *> functionArguments;
    for (auto argument : lambda->GetFunctionArguments())
      functionArguments.push_back(argument);

    assert(
        functionArguments.size() == 1 && functionArguments[0] == lambda->GetFunctionArguments()[0]);
  }

  {
    auto functionType = FunctionType::Create({}, { vt });

    auto lambda = lambda::node::create(
        &rvsdgModule.Rvsdg().GetRootRegion(),
        functionType,
        "f",
        linkage::external_linkage);

    auto nullaryNode = jlm::tests::create_testop(lambda->subregion(), {}, { vt });

    lambda->finalize({ nullaryNode });

    assert(lambda->GetFunctionArguments().empty());
  }

  {
    auto rvsdgImport = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), vt, "");

    auto functionType = FunctionType::Create({ vt, vt, vt }, { vt, vt });

    auto lambda = lambda::node::create(
        &rvsdgModule.Rvsdg().GetRootRegion(),
        functionType,
        "f",
        linkage::external_linkage);

    auto cv = lambda->AddContextVar(*rvsdgImport).inner;

    lambda->finalize({ lambda->GetFunctionArguments()[0], cv });

    std::vector<const jlm::rvsdg::output *> functionArguments;
    for (auto argument : lambda->GetFunctionArguments())
      functionArguments.push_back(argument);

    assert(functionArguments.size() == 3);
    assert(functionArguments[0] == lambda->GetFunctionArguments()[0]);
    assert(functionArguments[1] == lambda->GetFunctionArguments()[1]);
    assert(functionArguments[2] == lambda->GetFunctionArguments()[2]);
  }
}

static void
TestInvalidOperandRegion()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto functionType = FunctionType::Create({}, { vt });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto lambdaNode =
      lambda::node::create(&rvsdg->GetRootRegion(), functionType, "f", linkage::external_linkage);
  auto result = jlm::tests::create_testop(&rvsdg->GetRootRegion(), {}, { vt })[0];

  bool invalidRegionErrorCaught = false;
  try
  {
    lambdaNode->finalize({ result });
  }
  catch (jlm::util::error &)
  {
    invalidRegionErrorCaught = true;
  }

  assert(invalidRegionErrorCaught);
}

/**
 * Test lambda::node::RemoveLambdaInputsWhere()
 */
static void
TestRemoveLambdaInputsWhere()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = FunctionType::Create({}, { valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode =
      lambda::node::create(&rvsdg.GetRootRegion(), functionType, "f", linkage::external_linkage);

  auto lambdaBinder0 = lambdaNode->AddContextVar(*x);
  auto lambdaBinder1 = lambdaNode->AddContextVar(*x);
  lambdaNode->AddContextVar(*x);

  auto result = jlm::tests::SimpleNode::Create(
                    *lambdaNode->subregion(),
                    { lambdaBinder1.inner },
                    { valueType })
                    .output(0);

  lambdaNode->finalize({ result });

  // Act & Assert
  // Try to remove lambdaInput1 even though it is used
  auto numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::input & input)
      {
        return input.index() == lambdaBinder1.input->index();
      });
  assert(numRemovedInputs == 0);
  assert(lambdaNode->ninputs() == 3);
  assert(lambdaNode->GetContextVars().size() == 3);

  // Remove lambdaInput2
  numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::input & input)
      {
        return input.index() == 2;
      });
  assert(numRemovedInputs == 1);
  assert(lambdaNode->ninputs() == 2);
  assert(lambdaNode->GetContextVars().size() == 2);
  assert(lambdaNode->input(0) == lambdaBinder0.input);
  assert(lambdaNode->input(1) == lambdaBinder1.input);

  // Remove lambdaInput0
  numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::input & input)
      {
        return input.index() == 0;
      });
  assert(numRemovedInputs == 1);
  assert(lambdaNode->ninputs() == 1);
  assert(lambdaNode->GetContextVars().size() == 1);
  assert(lambdaNode->input(0) == lambdaBinder1.input);
  assert(lambdaBinder1.input->index() == 0);
  assert(lambdaBinder1.inner->index() == 0);
}

/**
 * Test lambda::node::PruneLambdaInputs()
 */
static void
TestPruneLambdaInputs()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = FunctionType::Create({}, { valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode =
      lambda::node::create(&rvsdg.GetRootRegion(), functionType, "f", linkage::external_linkage);

  lambdaNode->AddContextVar(*x);
  auto lambdaInput1 = lambdaNode->AddContextVar(*x);
  lambdaNode->AddContextVar(*x);

  auto result = jlm::tests::SimpleNode::Create(
                    *lambdaNode->subregion(),
                    { lambdaInput1.inner },
                    { valueType })
                    .output(0);

  lambdaNode->finalize({ result });

  // Act
  auto numRemovedInputs = lambdaNode->PruneLambdaInputs();

  // Assert
  assert(numRemovedInputs == 2);
  assert(lambdaNode->ninputs() == 1);
  assert(lambdaNode->GetContextVars().size() == 1);
  assert(lambdaNode->input(0) == lambdaInput1.input);
  assert(lambdaNode->GetContextVars()[0].inner == lambdaInput1.inner);
  assert(lambdaInput1.input->index() == 0);
  assert(lambdaInput1.inner->index() == 0);
}

static void
TestCallSummaryComputationDead()
{
  using namespace jlm;

  // Arrange
  auto vt = tests::valuetype::Create();
  auto functionType = jlm::llvm::FunctionType::Create({}, { vt });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::llvm::lambda::node::create(
      &rvsdg.GetRootRegion(),
      functionType,
      "f",
      jlm::llvm::linkage::external_linkage);

  auto result = tests::create_testop(lambdaNode->subregion(), {}, { vt })[0];

  lambdaNode->finalize({ result });

  // Act
  auto callSummary = lambdaNode->ComputeCallSummary();

  // Assert
  assert(callSummary->IsDead());

  assert(callSummary->IsExported() == false);
  assert(callSummary->IsOnlyExported() == false);
  assert(callSummary->GetRvsdgExport() == nullptr);
  assert(callSummary->HasOnlyDirectCalls() == false);
}

static void
TestCallSummaryComputationExport()
{
  using namespace jlm;

  // Arrange
  auto vt = tests::valuetype::Create();
  auto functionType = jlm::llvm::FunctionType::Create({}, { vt });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::llvm::lambda::node::create(
      &rvsdg.GetRootRegion(),
      functionType,
      "f",
      jlm::llvm::linkage::external_linkage);

  auto result = tests::create_testop(lambdaNode->subregion(), {}, { vt })[0];

  auto lambdaOutput = lambdaNode->finalize({ result });
  auto & rvsdgExport = jlm::llvm::GraphExport::Create(*lambdaOutput, "f");

  // Act
  auto callSummary = lambdaNode->ComputeCallSummary();

  // Assert
  assert(callSummary->IsExported());
  assert(callSummary->IsOnlyExported());
  assert(callSummary->GetRvsdgExport() == &rvsdgExport);

  assert(callSummary->IsDead() == false);
  assert(callSummary->HasOnlyDirectCalls() == false);
}

static void
TestCallSummaryComputationDirectCalls()
{
  using namespace jlm;

  // Arrange
  auto vt = tests::valuetype::Create();
  auto functionType = jlm::llvm::FunctionType::Create(
      { jlm::llvm::iostatetype::Create(), jlm::llvm::MemoryStateType::Create() },
      { vt, jlm::llvm::iostatetype::Create(), jlm::llvm::MemoryStateType::Create() });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto SetupLambdaX = [&]()
  {
    auto lambdaNode = jlm::llvm::lambda::node::create(
        &rvsdg.GetRootRegion(),
        functionType,
        "x",
        jlm::llvm::linkage::external_linkage);
    auto iOStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];

    auto result = tests::create_testop(lambdaNode->subregion(), {}, { vt })[0];

    return lambdaNode->finalize({ result, iOStateArgument, memoryStateArgument });
  };

  auto SetupLambdaY = [&](rvsdg::output & lambdaX)
  {
    auto lambdaNode = jlm::llvm::lambda::node::create(
        &rvsdg.GetRootRegion(),
        functionType,
        "y",
        jlm::llvm::linkage::external_linkage);
    auto iOStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto lambdaXCv = lambdaNode->AddContextVar(lambdaX).inner;

    auto callResults = jlm::llvm::CallNode::Create(
        lambdaXCv,
        functionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambdaNode->finalize(callResults);
    jlm::llvm::GraphExport::Create(*lambdaOutput, "y");

    return lambdaOutput;
  };

  auto SetupLambdaZ = [&](rvsdg::output & lambdaX, rvsdg::output & lambdaY)
  {
    auto lambdaNode = jlm::llvm::lambda::node::create(
        &rvsdg.GetRootRegion(),
        functionType,
        "y",
        jlm::llvm::linkage::external_linkage);
    auto iOStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto lambdaXCv = lambdaNode->AddContextVar(lambdaX).inner;
    auto lambdaYCv = lambdaNode->AddContextVar(lambdaY).inner;

    auto callXResults = jlm::llvm::CallNode::Create(
        lambdaXCv,
        functionType,
        { iOStateArgument, memoryStateArgument });
    auto callYResults =
        jlm::llvm::CallNode::Create(lambdaYCv, functionType, { callXResults[1], callXResults[2] });

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
      rvsdg::AssertGetOwnerNode<jlm::llvm::lambda::node>(*lambdaX).ComputeCallSummary();
  auto lambdaYCallSummary =
      rvsdg::AssertGetOwnerNode<jlm::llvm::lambda::node>(*lambdaY).ComputeCallSummary();
  auto lambdaZCallSummary =
      rvsdg::AssertGetOwnerNode<jlm::llvm::lambda::node>(*lambdaZ).ComputeCallSummary();

  // Assert
  assert(lambdaXCallSummary->HasOnlyDirectCalls());
  assert(lambdaXCallSummary->NumDirectCalls() == 2);
  assert(lambdaXCallSummary->IsDead() == false);
  assert(lambdaXCallSummary->IsExported() == false);
  assert(lambdaXCallSummary->IsOnlyExported() == false);

  assert(lambdaYCallSummary->IsDead() == false);
  assert(lambdaYCallSummary->HasOnlyDirectCalls() == false);
  assert(lambdaYCallSummary->NumDirectCalls() == 1);
  assert(lambdaYCallSummary->IsExported());
  assert(lambdaYCallSummary->IsOnlyExported() == false);

  assert(lambdaZCallSummary->IsDead() == false);
  assert(lambdaZCallSummary->HasOnlyDirectCalls() == false);
  assert(lambdaZCallSummary->NumDirectCalls() == 0);
  assert(lambdaZCallSummary->IsExported());
  assert(lambdaZCallSummary->IsOnlyExported());
}

static void
TestCallSummaryComputationIndirectCalls()
{
  // Arrange
  jlm::tests::IndirectCallTest1 test;
  test.module();

  // Act
  auto lambdaThreeCallSummary = test.GetLambdaThree().ComputeCallSummary();
  auto lambdaFourCallSummary = test.GetLambdaFour().ComputeCallSummary();
  auto lambdaIndcallCallSummary = test.GetLambdaIndcall().ComputeCallSummary();
  auto lambdaTestCallSummary = test.GetLambdaTest().ComputeCallSummary();

  // Assert
  assert(lambdaThreeCallSummary->HasOnlyDirectCalls() == false);
  assert(lambdaThreeCallSummary->NumDirectCalls() == 0);
  assert(lambdaThreeCallSummary->IsDead() == false);
  assert(lambdaThreeCallSummary->IsExported() == false);
  assert(lambdaThreeCallSummary->IsOnlyExported() == false);
  assert(lambdaThreeCallSummary->NumOtherUsers() == 1);

  assert(lambdaFourCallSummary->HasOnlyDirectCalls() == false);
  assert(lambdaFourCallSummary->NumDirectCalls() == 0);
  assert(lambdaFourCallSummary->IsDead() == false);
  assert(lambdaFourCallSummary->IsExported() == false);
  assert(lambdaFourCallSummary->IsOnlyExported() == false);
  assert(lambdaFourCallSummary->NumOtherUsers() == 1);

  assert(lambdaIndcallCallSummary->HasOnlyDirectCalls());
  assert(lambdaIndcallCallSummary->NumDirectCalls() == 2);
  assert(lambdaIndcallCallSummary->IsDead() == false);
  assert(lambdaIndcallCallSummary->IsExported() == false);
  assert(lambdaIndcallCallSummary->IsOnlyExported() == false);
  assert(lambdaIndcallCallSummary->NumOtherUsers() == 0);

  assert(lambdaTestCallSummary->HasOnlyDirectCalls() == false);
  assert(lambdaTestCallSummary->NumDirectCalls() == 0);
  assert(lambdaTestCallSummary->IsDead() == false);
  assert(lambdaTestCallSummary->IsExported());
  assert(lambdaTestCallSummary->IsOnlyExported());
  assert(lambdaTestCallSummary->NumOtherUsers() == 0);
}

static void
TestCallSummaryComputationFunctionPointerInDelta()
{
  using namespace jlm::llvm;

  // Arrange
  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto nf = rvsdg->GetNodeNormalForm(typeid(jlm::rvsdg::Operation));
  nf->set_mutable(false);

  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  auto lambdaNode =
      lambda::node::create(&rvsdg->GetRootRegion(), functionType, "f", linkage::external_linkage);
  lambdaNode->finalize({ lambdaNode->GetFunctionArguments()[0] });

  auto deltaNode = delta::node::Create(
      &rvsdg->GetRootRegion(),
      PointerType::Create(),
      "fp",
      linkage::external_linkage,
      "",
      false);
  auto argument = deltaNode->add_ctxvar(
      jlm::rvsdg::CreateOpNode<FunctionToPointerOperation>({ lambdaNode->output() }, functionType)
          .output(0));
  deltaNode->finalize(argument);

  GraphExport::Create(*deltaNode->output(), "fp");

  // Act
  auto callSummary = lambdaNode->ComputeCallSummary();

  // Assert
  assert(callSummary->NumOtherUsers() == 1);
  assert(callSummary->HasOnlyOtherUsages());
}

static void
TestCallSummaryComputationLambdaResult()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::rvsdg::Graph rvsdg;

  auto nf = rvsdg.GetNodeNormalForm(typeid(jlm::rvsdg::Operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto functionTypeG = FunctionType::Create({ valueType }, { valueType });
  auto functionTypeF = FunctionType::Create({ valueType }, { PointerType::Create() });

  auto lambdaNodeG =
      lambda::node::create(&rvsdg.GetRootRegion(), functionTypeG, "g", linkage::external_linkage);
  auto lambdaOutputG = lambdaNodeG->finalize({ lambdaNodeG->GetFunctionArguments()[0] });

  auto lambdaNodeF =
      lambda::node::create(&rvsdg.GetRootRegion(), functionTypeF, "f", linkage::external_linkage);
  auto lambdaGArgument = lambdaNodeF->AddContextVar(*lambdaOutputG).inner;
  auto lambdaOutputF = lambdaNodeF->finalize(
      { jlm::rvsdg::CreateOpNode<FunctionToPointerOperation>({ lambdaGArgument }, functionTypeG)
            .output(0) });

  GraphExport::Create(*lambdaOutputF, "f");

  // Act
  auto callSummary = lambdaNodeG->ComputeCallSummary();

  // Assert
  assert(callSummary->NumOtherUsers() == 1);
  assert(callSummary->HasOnlyOtherUsages());
}

static int
Test()
{
  TestArgumentIterators();
  TestInvalidOperandRegion();
  TestRemoveLambdaInputsWhere();
  TestPruneLambdaInputs();

  TestCallSummaryComputationDead();
  TestCallSummaryComputationExport();
  TestCallSummaryComputationDirectCalls();
  TestCallSummaryComputationIndirectCalls();
  TestCallSummaryComputationFunctionPointerInDelta();
  TestCallSummaryComputationLambdaResult();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestLambda", Test)
