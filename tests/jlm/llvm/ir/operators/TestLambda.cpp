/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <TestRvsdgs.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>

static void
TestArgumentIterators()
{
  using namespace jlm::llvm;

  jlm::tests::valuetype vt;
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  {
    FunctionType functionType({&vt}, {&vt});

    auto lambda = lambda::node::create(
      rvsdgModule.Rvsdg().root(),
      functionType,
      "f",
      linkage::external_linkage);
    lambda->finalize({lambda->fctargument(0)});

    std::vector<jlm::rvsdg::argument*> functionArguments;
    for (auto & argument : lambda->fctarguments())
      functionArguments.push_back(&argument);

    assert(functionArguments.size() == 1
           && functionArguments[0] == lambda->fctargument(0));
  }

  {
    FunctionType functionType({}, {&vt});

    auto lambda = lambda::node::create(
      rvsdgModule.Rvsdg().root(),
      functionType,
      "f",
      linkage::external_linkage);

    auto nullaryNode = jlm::tests::create_testop(lambda->subregion(), {}, {&vt});

    lambda->finalize({nullaryNode});

    assert(lambda->nfctarguments() == 0);
  }

  {
    auto rvsdgImport = rvsdgModule.Rvsdg().add_import({vt, ""});

    FunctionType functionType({&vt, &vt, &vt}, {&vt, &vt});

    auto lambda = lambda::node::create(
      rvsdgModule.Rvsdg().root(),
      functionType,
      "f",
      linkage::external_linkage);

    auto cv = lambda->add_ctxvar(rvsdgImport);

    lambda->finalize({lambda->fctargument(0), cv});

    std::vector<jlm::rvsdg::argument*> functionArguments;
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

  jlm::tests::valuetype vt;
  FunctionType functionType({}, {&vt});

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto lambdaNode = lambda::node::create(
    rvsdg->root(),
    functionType,
    "f",
    linkage::external_linkage);
  auto result = jlm::tests::create_testop(rvsdg->root(), {}, {&vt})[0];

  bool invalidRegionErrorCaught = false;
  try {
    lambdaNode->finalize({result});
  } catch (jlm::util::error&) {
    invalidRegionErrorCaught = true;
  }

  assert(invalidRegionErrorCaught);
}

static void
TestCallSummaryComputationDead()
{
  using namespace jlm;

  // Arrange
  tests::valuetype vt;
  jlm::llvm::FunctionType functionType({}, {&vt});

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::llvm::lambda::node::create(
    rvsdg.root(),
    functionType,
    "f",
    jlm::llvm::linkage::external_linkage);

  auto result = tests::create_testop(lambdaNode->subregion(), {}, {&vt})[0];

  lambdaNode->finalize({result});

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
  tests::valuetype vt;
  jlm::llvm::FunctionType functionType({}, {&vt});

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::llvm::lambda::node::create(
    rvsdg.root(),
    functionType,
    "f",
    jlm::llvm::linkage::external_linkage);

  auto result = tests::create_testop(lambdaNode->subregion(), {}, {&vt})[0];

  auto lambdaOutput = lambdaNode->finalize({result});
  auto rvsdgExport = rvsdg.add_export(lambdaOutput, {jlm::llvm::PointerType(), "f"});

  // Act
  auto callSummary = lambdaNode->ComputeCallSummary();

  // Assert
  assert(callSummary->IsExported());
  assert(callSummary->IsOnlyExported());
  assert(callSummary->GetRvsdgExport() == rvsdgExport);

  assert(callSummary->IsDead() == false);
  assert(callSummary->HasOnlyDirectCalls() == false);
}

static void
TestCallSummaryComputationDirectCalls()
{
  using namespace jlm;

  // Arrange
  tests::valuetype vt;
  jlm::llvm::iostatetype iOStateType;
  jlm::llvm::MemoryStateType memoryStateType;
  jlm::llvm::loopstatetype loopStateType;
  jlm::llvm::FunctionType functionType(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&vt, &iOStateType, &memoryStateType, &loopStateType});

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
    auto loopStateArgument = lambdaNode->fctargument(2);

    auto result = tests::create_testop(lambdaNode->subregion(), {}, {&vt})[0];

    return lambdaNode->finalize({result, iOStateArgument, memoryStateArgument, loopStateArgument});
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
    auto loopStateArgument = lambdaNode->fctargument(2);
    auto lambdaXCv = lambdaNode->add_ctxvar(&lambdaX);

    auto callResults = jlm::llvm::CallNode::Create(
      lambdaXCv,
      functionType,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    auto lambdaOutput = lambdaNode->finalize(callResults);
    rvsdg.add_export(lambdaOutput, {jlm::llvm::PointerType(), "y"});

    return lambdaOutput;
  };

  auto SetupLambdaZ = [&](
    jlm::llvm::lambda::output & lambdaX,
    jlm::llvm::lambda::output & lambdaY)
  {
    auto lambdaNode = jlm::llvm::lambda::node::create(
      rvsdg.root(),
      functionType,
      "y",
      jlm::llvm::linkage::external_linkage);
    auto iOStateArgument = lambdaNode->fctargument(0);
    auto memoryStateArgument = lambdaNode->fctargument(1);
    auto loopStateArgument = lambdaNode->fctargument(2);
    auto lambdaXCv = lambdaNode->add_ctxvar(&lambdaX);
    auto lambdaYCv = lambdaNode->add_ctxvar(&lambdaY);

    auto callXResults = jlm::llvm::CallNode::Create(
      lambdaXCv,
      functionType,
      {iOStateArgument, memoryStateArgument, loopStateArgument});
    auto callYResults = jlm::llvm::CallNode::Create(
      lambdaYCv,
      functionType,
      {callXResults[1], callXResults[2], callXResults[3]});

    auto result = tests::create_testop(
      lambdaNode->subregion(),
      {callXResults[0], callYResults[0]},
      {&vt})[0];

    auto lambdaOutput = lambdaNode->finalize({result, callYResults[1], callYResults[2], callYResults[3]});
    rvsdg.add_export(lambdaOutput, {jlm::llvm::PointerType(), "z"});

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

  jlm::tests::valuetype valueType;
  FunctionType functionType({&valueType}, {&valueType});

  auto lambdaNode = lambda::node::create(
    rvsdg->root(),
    functionType,
    "f",
    linkage::external_linkage);
  lambdaNode->finalize({lambdaNode->fctargument(0)});

  auto deltaNode = delta::node::Create(
    rvsdg->root(),
    PointerType(),
    "fp",
    linkage::external_linkage,
    "",
    false);
  auto argument = deltaNode->add_ctxvar(lambdaNode->output());
  deltaNode->finalize(argument);

  rvsdg->add_export(deltaNode->output(), {PointerType(), "fp"});

  // Act
  auto callSummary = lambdaNode->ComputeCallSummary();

  // Assert
  assert(callSummary->NumOtherUsers() == 1);
  assert(callSummary->HasOnlyOtherUsages());
}

static int
Test()
{
  TestArgumentIterators();
  TestInvalidOperandRegion();

  TestCallSummaryComputationDead();
  TestCallSummaryComputationExport();
  TestCallSummaryComputationDirectCalls();
  TestCallSummaryComputationIndirectCalls();
  TestCallSummaryComputationFunctionPointerInDelta();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestLambda", Test)
