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
        rvsdgModule.Rvsdg().root(),
        functionType,
        "f",
        linkage::external_linkage);
    lambda->finalize({ lambda->fctargument(0) });

    std::vector<jlm::rvsdg::argument *> functionArguments;
    for (auto & argument : lambda->fctarguments())
      functionArguments.push_back(&argument);

    assert(functionArguments.size() == 1 && functionArguments[0] == lambda->fctargument(0));
  }

  {
    auto functionType = FunctionType::Create({}, { vt });

    auto lambda = lambda::node::create(
        rvsdgModule.Rvsdg().root(),
        functionType,
        "f",
        linkage::external_linkage);

    auto nullaryNode = jlm::tests::create_testop(lambda->subregion(), {}, { vt });

    lambda->finalize({ nullaryNode });

    assert(lambda->nfctarguments() == 0);
  }

  {
    auto rvsdgImport = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), vt, "");

    auto functionType = FunctionType::Create({ vt, vt, vt }, { vt, vt });

    auto lambda = lambda::node::create(
        rvsdgModule.Rvsdg().root(),
        functionType,
        "f",
        linkage::external_linkage);

    auto cv = lambda->add_ctxvar(rvsdgImport);

    lambda->finalize({ lambda->fctargument(0), cv });

    std::vector<jlm::rvsdg::argument *> functionArguments;
    for (auto & argument : lambda->fctarguments())
      functionArguments.push_back(&argument);

    assert(functionArguments.size() == 3);
    assert(functionArguments[0] == lambda->fctargument(0));
    assert(functionArguments[1] == lambda->fctargument(1));
    assert(functionArguments[2] == lambda->fctargument(2));
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
      lambda::node::create(rvsdg->root(), functionType, "f", linkage::external_linkage);
  auto result = jlm::tests::create_testop(rvsdg->root(), {}, { vt })[0];

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
      lambda::node::create(rvsdg.root(), functionType, "f", linkage::external_linkage);

  auto lambdaInput0 = lambdaNode->add_ctxvar(x)->input();
  auto lambdaInput1 = lambdaNode->add_ctxvar(x)->input();
  lambdaNode->add_ctxvar(x)->input();

  auto result = jlm::tests::SimpleNode::Create(
                    *lambdaNode->subregion(),
                    { lambdaInput1->argument() },
                    { valueType })
                    .output(0);

  lambdaNode->finalize({ result });

  // Act & Assert
  // Try to remove lambdaInput1 even though it is used
  auto numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const lambda::cvinput & input)
      {
        return input.index() == lambdaInput1->index();
      });
  assert(numRemovedInputs == 0);
  assert(lambdaNode->ninputs() == 3);
  assert(lambdaNode->ncvarguments() == 3);

  // Remove lambdaInput2
  numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const lambda::cvinput & input)
      {
        return input.index() == 2;
      });
  assert(numRemovedInputs == 1);
  assert(lambdaNode->ninputs() == 2);
  assert(lambdaNode->ncvarguments() == 2);
  assert(lambdaNode->input(0) == lambdaInput0);
  assert(lambdaNode->input(1) == lambdaInput1);

  // Remove lambdaInput0
  numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const lambda::cvinput & input)
      {
        return input.index() == 0;
      });
  assert(numRemovedInputs == 1);
  assert(lambdaNode->ninputs() == 1);
  assert(lambdaNode->ncvarguments() == 1);
  assert(lambdaNode->input(0) == lambdaInput1);
  assert(lambdaInput1->index() == 0);
  assert(lambdaInput1->argument()->index() == 0);
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
      lambda::node::create(rvsdg.root(), functionType, "f", linkage::external_linkage);

  lambdaNode->add_ctxvar(x)->input();
  auto lambdaInput1 = lambdaNode->add_ctxvar(x)->input();
  lambdaNode->add_ctxvar(x)->input();

  auto result = jlm::tests::SimpleNode::Create(
                    *lambdaNode->subregion(),
                    { lambdaInput1->argument() },
                    { valueType })
                    .output(0);

  lambdaNode->finalize({ result });

  // Act
  auto numRemovedInputs = lambdaNode->PruneLambdaInputs();

  // Assert
  assert(numRemovedInputs == 2);
  assert(lambdaNode->ninputs() == 1);
  assert(lambdaNode->ncvarguments() == 1);
  assert(lambdaNode->input(0) == lambdaInput1);
  assert(lambdaNode->cvargument(0) == lambdaInput1->argument());
  assert(lambdaInput1->index() == 0);
  assert(lambdaInput1->argument()->index() == 0);
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
      rvsdg.root(),
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
      rvsdg.root(),
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
        rvsdg.root(),
        functionType,
        "x",
        jlm::llvm::linkage::external_linkage);
    auto iOStateArgument = lambdaNode->fctargument(0);
    auto memoryStateArgument = lambdaNode->fctargument(1);

    auto result = tests::create_testop(lambdaNode->subregion(), {}, { vt })[0];

    return lambdaNode->finalize({ result, iOStateArgument, memoryStateArgument });
  };

  auto SetupLambdaY = [&](jlm::llvm::lambda::output & lambdaX)
  {
    auto lambdaNode = jlm::llvm::lambda::node::create(
        rvsdg.root(),
        functionType,
        "y",
        jlm::llvm::linkage::external_linkage);
    auto iOStateArgument = lambdaNode->fctargument(0);
    auto memoryStateArgument = lambdaNode->fctargument(1);
    auto lambdaXCv = lambdaNode->add_ctxvar(&lambdaX);

    auto callResults = jlm::llvm::CallNode::Create(
        lambdaXCv,
        functionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambdaNode->finalize(callResults);
    jlm::llvm::GraphExport::Create(*lambdaOutput, "y");

    return lambdaOutput;
  };

  auto SetupLambdaZ = [&](jlm::llvm::lambda::output & lambdaX, jlm::llvm::lambda::output & lambdaY)
  {
    auto lambdaNode = jlm::llvm::lambda::node::create(
        rvsdg.root(),
        functionType,
        "y",
        jlm::llvm::linkage::external_linkage);
    auto iOStateArgument = lambdaNode->fctargument(0);
    auto memoryStateArgument = lambdaNode->fctargument(1);
    auto lambdaXCv = lambdaNode->add_ctxvar(&lambdaX);
    auto lambdaYCv = lambdaNode->add_ctxvar(&lambdaY);

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
  auto lambdaXCallSummary = lambdaX->node()->ComputeCallSummary();
  auto lambdaYCallSummary = lambdaY->node()->ComputeCallSummary();
  auto lambdaZCallSummary = lambdaZ->node()->ComputeCallSummary();

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

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  auto lambdaNode =
      lambda::node::create(rvsdg->root(), functionType, "f", linkage::external_linkage);
  lambdaNode->finalize({ lambdaNode->fctargument(0) });

  auto deltaNode = delta::node::Create(
      rvsdg->root(),
      PointerType::Create(),
      "fp",
      linkage::external_linkage,
      "",
      false);
  auto argument = deltaNode->add_ctxvar(lambdaNode->output());
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
  jlm::rvsdg::graph rvsdg;

  auto nf = rvsdg.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto functionTypeG = FunctionType::Create({ valueType }, { valueType });
  auto functionTypeF = FunctionType::Create({ valueType }, { PointerType::Create() });

  auto lambdaNodeG =
      lambda::node::create(rvsdg.root(), functionTypeG, "g", linkage::external_linkage);
  auto lambdaOutputG = lambdaNodeG->finalize({ lambdaNodeG->fctargument(0) });

  auto lambdaNodeF =
      lambda::node::create(rvsdg.root(), functionTypeF, "f", linkage::external_linkage);
  auto lambdaGArgument = lambdaNodeF->add_ctxvar(lambdaOutputG);
  auto lambdaOutputF = lambdaNodeF->finalize({ lambdaGArgument });

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
