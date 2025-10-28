/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunInvariantValueRedirection(jlm::llvm::RvsdgModule & rvsdgModule)
{
  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::InvariantValueRedirection invariantValueRedirection;
  invariantValueRedirection.Run(rvsdgModule, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);
}

static void
TestGamma()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, valueType },
      { valueType, valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto c = lambdaNode->GetFunctionArguments()[0];
  auto x = lambdaNode->GetFunctionArguments()[1];
  auto y = lambdaNode->GetFunctionArguments()[2];

  auto gammaNode1 = jlm::rvsdg::GammaNode::create(c, 2);
  auto gammaInput1 = gammaNode1->AddEntryVar(c);
  auto gammaInput2 = gammaNode1->AddEntryVar(x);
  auto gammaInput3 = gammaNode1->AddEntryVar(y);

  auto gammaNode2 = jlm::rvsdg::GammaNode::create(gammaInput1.branchArgument[0], 2);
  auto gammaInput4 = gammaNode2->AddEntryVar(gammaInput2.branchArgument[0]);
  auto gammaInput5 = gammaNode2->AddEntryVar(gammaInput3.branchArgument[0]);
  gammaNode2->AddExitVar(gammaInput4.branchArgument);
  gammaNode2->AddExitVar(gammaInput5.branchArgument);

  gammaNode1->AddExitVar({ gammaNode2->output(0), gammaInput2.branchArgument[1] });
  gammaNode1->AddExitVar({ gammaNode2->output(1), gammaInput3.branchArgument[1] });

  auto lambdaOutput = lambdaNode->finalize({ gammaNode1->output(0), gammaNode1->output(1) });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  assert(lambdaNode->GetFunctionResults()[0]->origin() == x);
  assert(lambdaNode->GetFunctionResults()[1]->origin() == y);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Gamma", TestGamma)

static void
TestTheta()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = IOStateType::Create();
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, ioStateType },
      { controlType, valueType, ioStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto c = lambdaNode->GetFunctionArguments()[0];
  auto x = lambdaNode->GetFunctionArguments()[1];
  auto l = lambdaNode->GetFunctionArguments()[2];

  auto thetaNode1 = jlm::rvsdg::ThetaNode::create(lambdaNode->subregion());
  auto thetaVar1 = thetaNode1->AddLoopVar(c);
  auto thetaVar2 = thetaNode1->AddLoopVar(x);
  auto thetaVar3 = thetaNode1->AddLoopVar(l);

  auto thetaNode2 = jlm::rvsdg::ThetaNode::create(thetaNode1->subregion());
  auto thetaVar4 = thetaNode2->AddLoopVar(thetaVar1.pre);
  thetaNode2->AddLoopVar(thetaVar2.pre);
  auto thetaVar5 = thetaNode2->AddLoopVar(thetaVar3.pre);
  thetaNode2->set_predicate(thetaVar4.pre);

  thetaVar3.post->divert_to(thetaVar5.output);
  thetaNode1->set_predicate(thetaVar1.pre);

  auto lambdaOutput =
      lambdaNode->finalize({ thetaVar1.output, thetaVar2.output, thetaVar3.output });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  assert(lambdaNode->GetFunctionResults()[0]->origin() == c);
  assert(lambdaNode->GetFunctionResults()[1]->origin() == x);
  assert(lambdaNode->GetFunctionResults()[2]->origin() == thetaVar3.output);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Theta", TestTheta)

static void
TestCall()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionTypeTest1 = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, valueType, ioStateType, memoryStateType },
      { valueType, valueType, ioStateType, memoryStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  jlm::rvsdg::Output * lambdaOutputTest1 = nullptr;
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionTypeTest1, "test1", Linkage::externalLinkage));

    auto controlArgument = lambdaNode->GetFunctionArguments()[0];
    auto xArgument = lambdaNode->GetFunctionArguments()[1];
    auto yArgument = lambdaNode->GetFunctionArguments()[2];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[3];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[4];

    auto gammaNode = jlm::rvsdg::GammaNode::create(controlArgument, 2);
    auto gammaInputX = gammaNode->AddEntryVar(xArgument);
    auto gammaInputY = gammaNode->AddEntryVar(yArgument);
    auto gammaInputIOState = gammaNode->AddEntryVar(ioStateArgument);
    auto gammaInputMemoryState = gammaNode->AddEntryVar(memoryStateArgument);
    auto gammaOutputX =
        gammaNode->AddExitVar({ gammaInputY.branchArgument[0], gammaInputY.branchArgument[1] });
    auto gammaOutputY =
        gammaNode->AddExitVar({ gammaInputX.branchArgument[0], gammaInputX.branchArgument[1] });
    auto gammaOutputIOState = gammaNode->AddExitVar(
        { gammaInputIOState.branchArgument[0], gammaInputIOState.branchArgument[1] });
    auto gammaOutputMemoryState = gammaNode->AddExitVar(
        { gammaInputMemoryState.branchArgument[0], gammaInputMemoryState.branchArgument[1] });

    lambdaOutputTest1 = lambdaNode->finalize({ gammaOutputX.output,
                                               gammaOutputY.output,
                                               gammaOutputIOState.output,
                                               gammaOutputMemoryState.output });
  }

  jlm::rvsdg::Output * lambdaOutputTest2 = nullptr;
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { valueType, valueType, ioStateType, memoryStateType },
        { valueType, valueType, ioStateType, memoryStateType });

    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test2", Linkage::externalLinkage));
    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto yArgument = lambdaNode->GetFunctionArguments()[1];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[3];
    auto lambdaArgumentTest1 = lambdaNode->AddContextVar(*lambdaOutputTest1).inner;

    auto controlResult = jlm::rvsdg::control_constant(lambdaNode->subregion(), 2, 0);

    auto & callNode = CallOperation::CreateNode(
        lambdaArgumentTest1,
        functionTypeTest1,
        { controlResult, xArgument, yArgument, ioStateArgument, memoryStateArgument });

    lambdaOutputTest2 = lambdaNode->finalize(outputs(&callNode));
    jlm::rvsdg::GraphExport::Create(*lambdaOutputTest2, "test2");
  }

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  auto & lambdaNode = jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaOutputTest2);
  assert(lambdaNode.GetFunctionResults().size() == 4);
  assert(lambdaNode.GetFunctionResults()[0]->origin() == lambdaNode.GetFunctionArguments()[1]);
  assert(lambdaNode.GetFunctionResults()[1]->origin() == lambdaNode.GetFunctionArguments()[0]);
  assert(lambdaNode.GetFunctionResults()[2]->origin() == lambdaNode.GetFunctionArguments()[2]);
  assert(lambdaNode.GetFunctionResults()[3]->origin() == lambdaNode.GetFunctionArguments()[3]);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Call", TestCall)

static void
TestCallWithMemoryStateNodes()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionTypeTest1 = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, ioStateType, memoryStateType },
      { valueType, ioStateType, memoryStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  jlm::rvsdg::Output * lambdaOutputTest1 = nullptr;
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionTypeTest1, "test1", Linkage::externalLinkage));

    auto controlArgument = lambdaNode->GetFunctionArguments()[0];
    auto xArgument = lambdaNode->GetFunctionArguments()[1];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[3];

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 0, 1 });

    auto gammaNode = jlm::rvsdg::GammaNode::create(controlArgument, 2);

    auto gammaInputX = gammaNode->AddEntryVar(xArgument);
    auto gammaInputMemoryState1 = gammaNode->AddEntryVar(lambdaEntrySplitNode.output(0));
    auto gammaInputMemoryState2 = gammaNode->AddEntryVar(lambdaEntrySplitNode.output(1));

    auto gammaOutputX = gammaNode->AddExitVar(gammaInputX.branchArgument);
    auto gammaOutputMemoryState1 = gammaNode->AddExitVar(gammaInputMemoryState1.branchArgument);
    auto gammaOutputMemoryState2 = gammaNode->AddExitVar(gammaInputMemoryState2.branchArgument);

    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        { gammaOutputMemoryState1.output, gammaOutputMemoryState2.output },
        { 0, 1 });

    lambdaOutputTest1 = lambdaNode->finalize(
        { gammaOutputX.output, ioStateArgument, lambdaExitMergeNode.output(0) });
  }

  jlm::rvsdg::Output * lambdaOutputTest2 = nullptr;
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { valueType, ioStateType, memoryStateType },
        { valueType, ioStateType, memoryStateType });

    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test2", Linkage::externalLinkage));
    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto lambdaArgumentTest1 = lambdaNode->AddContextVar(*lambdaOutputTest1).inner;

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 0, 1 });

    auto & callEntryMergeNode = CallEntryMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(&lambdaEntrySplitNode),
        { 0, 1 });

    auto controlResult = jlm::rvsdg::control_constant(lambdaNode->subregion(), 2, 0);

    auto & callNode = CallOperation::CreateNode(
        lambdaArgumentTest1,
        functionTypeTest1,
        { controlResult, xArgument, ioStateArgument, callEntryMergeNode.output(0) });

    auto & callExitSplitNode = CallExitMemoryStateSplitOperation::CreateNode(
        CallOperation::GetMemoryStateOutput(callNode),
        { 1, 0 });

    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(&callExitSplitNode),
        { 1, 0 });

    lambdaOutputTest2 = lambdaNode->finalize({ callNode.output(0),
                                               &CallOperation::GetIOStateOutput(callNode),
                                               lambdaExitMergeNode.output(0) });
    jlm::rvsdg::GraphExport::Create(*lambdaOutputTest2, "test2");
  }

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  auto & lambdaNode = jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaOutputTest2);
  assert(lambdaNode.GetFunctionResults().size() == 3);
  assert(lambdaNode.GetFunctionResults()[0]->origin() == lambdaNode.GetFunctionArguments()[0]);
  assert(lambdaNode.GetFunctionResults()[1]->origin() == lambdaNode.GetFunctionArguments()[1]);

  auto lambdaEntrySplit = GetMemoryStateEntrySplit(lambdaNode);
  auto lambdaExitMerge = GetMemoryStateExitMerge(lambdaNode);

  assert(lambdaEntrySplit->noutputs() == 2);
  assert(lambdaExitMerge->ninputs() == 2);
  assert(lambdaExitMerge->input(0)->origin() == lambdaEntrySplit->output(1));
  assert(lambdaExitMerge->input(1)->origin() == lambdaEntrySplit->output(0));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/InvariantValueRedirectionTests-CallWithMemoryStateNodes",
    TestCallWithMemoryStateNodes)

static void
TestCallWithMissingMemoryStateNodes()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::tests::ValueType::Create();
  auto int32Type = BitType::Create(32);
  auto functionType = FunctionType::Create(
      { valueType, ioStateType, memoryStateType },
      { int32Type, ioStateType, memoryStateType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  Output * lambdaOutputTest1 = nullptr;
  {
    auto lambdaNode = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[2];

    auto & zeroNode = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 0);
    auto & oneNode = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 1);
    auto allocaResults = AllocaOperation::create(valueType, oneNode.output(0), 4);

    auto & storeNode = StoreNonVolatileOperation::CreateNode(
        *allocaResults[0],
        *xArgument,
        { memoryStateArgument },
        4);

    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        { storeNode.output(0) },
        { 0 });

    lambdaOutputTest1 = lambdaNode->finalize(
        { zeroNode.output(0), ioStateArgument, lambdaExitMergeNode.output(0) });
  }

  Output * lambdaOutputTest2 = nullptr;
  {
    auto lambdaNode = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test2", Linkage::externalLinkage));
    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto lambdaArgumentTest = lambdaNode->AddContextVar(*lambdaOutputTest1).inner;

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 0 });

    auto & callEntryMergeNode = CallEntryMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(&lambdaEntrySplitNode),
        { 0 });

    auto & callNode = CallOperation::CreateNode(
        lambdaArgumentTest,
        functionType,
        { xArgument, ioStateArgument, callEntryMergeNode.output(0) });

    auto & callExitSplitNode = CallExitMemoryStateSplitOperation::CreateNode(
        CallOperation::GetMemoryStateOutput(callNode),
        { 0 });

    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(&callExitSplitNode),
        { 0 });

    lambdaOutputTest2 = lambdaNode->finalize({ callNode.output(0),
                                               &CallOperation::GetIOStateOutput(callNode),
                                               lambdaExitMergeNode.output(0) });
    GraphExport::Create(*lambdaOutputTest2, "test2");
  }

  std::cout << view(&rvsdg.GetRootRegion()) << std::flush;

  // Act
  RunInvariantValueRedirection(*rvsdgModule);
  std::cout << view(&rvsdg.GetRootRegion()) << std::flush;

  // Assert
  // Nothing should have been redirected
  const auto & lambdaNode1 = AssertGetOwnerNode<LambdaNode>(*lambdaOutputTest1);
  const auto lambdaEntrySplit1 = GetMemoryStateEntrySplit(lambdaNode1);
  const auto lambdaExitMerge1 = GetMemoryStateExitMerge(lambdaNode1);
  assert(lambdaEntrySplit1 == nullptr);
  assert(lambdaExitMerge1->ninputs() == 1);

  const auto & lambdaNode2 = AssertGetOwnerNode<LambdaNode>(*lambdaOutputTest2);
  const auto lambdaEntrySplit2 = GetMemoryStateEntrySplit(lambdaNode2);
  const auto lambdaExitMerge2 = GetMemoryStateExitMerge(lambdaNode2);
  assert(lambdaEntrySplit2->noutputs() == 1);
  assert(lambdaExitMerge2->ninputs() == 1);
  const auto & [callExitSplitNode, _] =
      TryGetSimpleNodeAndOptionalOp<CallExitMemoryStateSplitOperation>(
          *lambdaExitMerge2->input(0)->origin());
  assert(callExitSplitNode->noutputs() == 1);
  const auto & [callNode, calOperation] =
      TryGetSimpleNodeAndOptionalOp<CallOperation>(*callExitSplitNode->input(0)->origin());
  assert(callNode->noutputs() == 3);
  assert(callNode->ninputs() == 4);
  const auto & memoryStateInput = CallOperation::GetMemoryStateInput(*callNode);
  const auto & [callEntryMergeNode, callEntryMergeOperation] =
      TryGetSimpleNodeAndOptionalOp<CallEntryMemoryStateMergeOperation>(*memoryStateInput.origin());
  assert(callEntryMergeNode->ninputs() == 1);
  assert(callEntryMergeNode->input(0)->origin() == lambdaEntrySplit2->output(0));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/InvariantValueRedirectionTests-CallWithMissingMemoryStateNodes",
    TestCallWithMissingMemoryStateNodes)

static void
TestLambdaCallArgumentMismatch()
{
  // Arrange
  jlm::tests::LambdaCallArgumentMismatch test;

  // Act
  RunInvariantValueRedirection(test.module());

  // Assert
  auto & callNode = test.GetCall();
  auto & lambdaNode = test.GetLambdaMain();

  assert(lambdaNode.GetFunctionResults().size() == 3);
  assert(lambdaNode.GetFunctionResults().size() == callNode.noutputs());
  assert(lambdaNode.GetFunctionResults()[0]->origin() == callNode.output(0));
  assert(lambdaNode.GetFunctionResults()[1]->origin() == callNode.output(1));
  assert(lambdaNode.GetFunctionResults()[2]->origin() == callNode.output(2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/InvariantValueRedirectionTests-LambdaCallArgumentMismatch",
    TestLambdaCallArgumentMismatch)

static void
testThetaGammaRedirection()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto valueType = ValueType::Create();
  auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create({ valueType, valueType }, { valueType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto functionArgument0 = lambdaNode->GetFunctionArguments()[0];
  auto functionArgument1 = lambdaNode->GetFunctionArguments()[1];

  auto thetaNode = ThetaNode::create(lambdaNode->subregion());
  auto loopVar0 = thetaNode->AddLoopVar(functionArgument0);
  auto loopVar1 = thetaNode->AddLoopVar(functionArgument1);

  auto dummyNodeTheta = TestOperation::create(thetaNode->subregion(), {}, { valueType });

  auto predicate = TestOperation::create(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode = GammaNode::create(predicate, 2);
  auto entryVar0 = gammaNode->AddEntryVar(loopVar0.pre);
  auto entryVar1 = gammaNode->AddEntryVar(dummyNodeTheta->output(0));

  auto dummyNodeGamma0 = TestOperation::create(gammaNode->subregion(0), {}, { valueType });
  auto dummyNodeGamma1 = TestOperation::create(gammaNode->subregion(1), {}, { valueType });

  auto controlConstant0 =
      ControlConstantOperation::create(gammaNode->subregion(0), ControlValueRepresentation(0, 2));
  auto controlConstant1 =
      ControlConstantOperation::create(gammaNode->subregion(1), ControlValueRepresentation(1, 2));

  auto controlExitVar = gammaNode->AddExitVar({ controlConstant0, controlConstant1 });
  auto exitVar0 =
      gammaNode->AddExitVar({ dummyNodeGamma0->output(0), entryVar0.branchArgument[1] });
  auto exitVar1 =
      gammaNode->AddExitVar({ entryVar1.branchArgument[0], dummyNodeGamma1->output(0) });

  thetaNode->predicate()->divert_to(controlExitVar.output);
  loopVar0.post->divert_to(exitVar0.output);
  loopVar1.post->divert_to(exitVar1.output);

  auto lambdaOutput = lambdaNode->finalize({ loopVar1.output });

  GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  // We expect that the post value of both loop variables does not originate from the gamma any
  // longer.
  auto loopVars = thetaNode->GetLoopVars();
  assert(loopVars.size() == 2);

  // Loop variable 0 was dead after the loop, which means it is irrelevant what happens to it in
  // the last iteration of the loop. As the loop predicate originates from a control constant in
  // one of the gamma nodes' subregions, the loop variables' value is always the same as the one
  // from the gamma subregion with control constant 1 (i.e. loop repetition). This means we could
  // redirect the loop variable from the gamma to the respective entry variables' origin.
  assert(loopVars[0].post->origin() == loopVars[0].pre);

  // Loop variable 1 was dead at the beginning of each loop iteration, which means it is irrelevant
  // what happens to it except in the last iteration of the loop. As the loop predicate originates
  // from a control constant in a one of the gamma nodes' subregions, the loop variables' value is
  // always the same as the one from the gamma subregion with control constant 0 (i.e. loop exit).
  // This means we could redirect the loop variable from the gamma to the respective entry
  // variables' origin.
  assert(loopVars[1].post->origin() == dummyNodeTheta->output(0));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/InvariantValueRedirectionTests-testThetaGammaRedirection",
    testThetaGammaRedirection)
