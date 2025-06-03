/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
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

static int
TestGamma()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, valueType },
      { valueType, valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

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

  GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  assert(lambdaNode->GetFunctionResults()[0]->origin() == x);
  assert(lambdaNode->GetFunctionResults()[1]->origin() == y);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Gamma", TestGamma)

static int
TestTheta()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = IOStateType::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, ioStateType },
      { controlType, valueType, ioStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

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

  GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  assert(lambdaNode->GetFunctionResults()[0]->origin() == c);
  assert(lambdaNode->GetFunctionResults()[1]->origin() == x);
  assert(lambdaNode->GetFunctionResults()[2]->origin() == thetaVar3.output);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Theta", TestTheta)

static int
TestCall()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionTypeTest1 = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, valueType, ioStateType, memoryStateType },
      { valueType, valueType, ioStateType, memoryStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  jlm::rvsdg::Output * lambdaOutputTest1;
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionTypeTest1, "test1", linkage::external_linkage));

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

  jlm::rvsdg::Output * lambdaOutputTest2;
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { valueType, valueType, ioStateType, memoryStateType },
        { valueType, valueType, ioStateType, memoryStateType });

    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test2", linkage::external_linkage));
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
    GraphExport::Create(*lambdaOutputTest2, "test2");
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

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Call", TestCall)

static int
TestCallWithMemoryStateNodes()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionTypeTest1 = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, ioStateType, memoryStateType },
      { valueType, ioStateType, memoryStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  jlm::rvsdg::Output * lambdaOutputTest1;
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionTypeTest1, "test1", linkage::external_linkage));

    auto controlArgument = lambdaNode->GetFunctionArguments()[0];
    auto xArgument = lambdaNode->GetFunctionArguments()[1];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[3];

    auto lambdaEntrySplitResults =
        LambdaEntryMemoryStateSplitOperation::Create(*memoryStateArgument, 2);

    auto gammaNode = jlm::rvsdg::GammaNode::create(controlArgument, 2);

    auto gammaInputX = gammaNode->AddEntryVar(xArgument);
    auto gammaInputMemoryState1 = gammaNode->AddEntryVar(lambdaEntrySplitResults[0]);
    auto gammaInputMemoryState2 = gammaNode->AddEntryVar(lambdaEntrySplitResults[1]);

    auto gammaOutputX = gammaNode->AddExitVar(gammaInputX.branchArgument);
    auto gammaOutputMemoryState1 = gammaNode->AddExitVar(gammaInputMemoryState2.branchArgument);
    auto gammaOutputMemoryState2 = gammaNode->AddExitVar(gammaInputMemoryState1.branchArgument);

    auto & lambdaExitMergeResult = LambdaExitMemoryStateMergeOperation::Create(
        *lambdaNode->subregion(),
        { gammaOutputMemoryState1.output, gammaOutputMemoryState2.output });

    lambdaOutputTest1 =
        lambdaNode->finalize({ gammaOutputX.output, ioStateArgument, &lambdaExitMergeResult });
  }

  jlm::rvsdg::Output * lambdaOutputTest2;
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { valueType, ioStateType, memoryStateType },
        { valueType, ioStateType, memoryStateType });

    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test2", linkage::external_linkage));
    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto lambdaArgumentTest1 = lambdaNode->AddContextVar(*lambdaOutputTest1).inner;

    auto lambdaEntrySplitResults =
        LambdaEntryMemoryStateSplitOperation::Create(*memoryStateArgument, 2);

    auto & callEntryMergeResult = CallEntryMemoryStateMergeOperation::Create(
        *lambdaNode->subregion(),
        lambdaEntrySplitResults);

    auto controlResult = jlm::rvsdg::control_constant(lambdaNode->subregion(), 2, 0);

    auto & callNode = CallOperation::CreateNode(
        lambdaArgumentTest1,
        functionTypeTest1,
        { controlResult, xArgument, ioStateArgument, &callEntryMergeResult });

    auto callExitSplitResults =
        CallExitMemoryStateSplitOperation::Create(CallOperation::GetMemoryStateOutput(callNode), 2);

    auto & lambdaExitMergeResult =
        LambdaExitMemoryStateMergeOperation::Create(*lambdaNode->subregion(), callExitSplitResults);

    lambdaOutputTest2 = lambdaNode->finalize(
        { callNode.output(0), &CallOperation::GetIOStateOutput(callNode), &lambdaExitMergeResult });
    GraphExport::Create(*lambdaOutputTest2, "test2");
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

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/InvariantValueRedirectionTests-CallWithMemoryStateNodes",
    TestCallWithMemoryStateNodes)

static int
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

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/InvariantValueRedirectionTests-LambdaCallArgumentMismatch",
    TestLambdaCallArgumentMismatch)
