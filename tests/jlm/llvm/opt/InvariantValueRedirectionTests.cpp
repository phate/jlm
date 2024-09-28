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
  invariantValueRedirection.run(rvsdgModule, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);
}

static int
TestGamma()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto controlType = jlm::rvsdg::ctltype::Create(2);
  auto functionType =
      FunctionType::Create({ controlType, valueType, valueType }, { valueType, valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode =
      lambda::node::create(rvsdg.root(), functionType, "test", linkage::external_linkage);

  auto c = lambdaNode->fctargument(0);
  auto x = lambdaNode->fctargument(1);
  auto y = lambdaNode->fctargument(2);

  auto gammaNode1 = jlm::rvsdg::GammaNode::create(c, 2);
  auto gammaInput1 = gammaNode1->AddEntryVar(c);
  auto gammaInput2 = gammaNode1->AddEntryVar(x);
  auto gammaInput3 = gammaNode1->AddEntryVar(y);

  auto gammaNode2 = jlm::rvsdg::GammaNode::create(gammaInput1.branches[0], 2);
  auto gammaInput4 = gammaNode2->AddEntryVar(gammaInput2.branches[0]);
  auto gammaInput5 = gammaNode2->AddEntryVar(gammaInput3.branches[0]);
  gammaNode2->add_exitvar(gammaInput4.branches);
  gammaNode2->add_exitvar(gammaInput5.branches);

  gammaNode1->add_exitvar({ gammaNode2->output(0), gammaInput2.branches[1] });
  gammaNode1->add_exitvar({ gammaNode2->output(1), gammaInput3.branches[1] });

  auto lambdaOutput = lambdaNode->finalize({ gammaNode1->output(0), gammaNode1->output(1) });

  GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  assert(lambdaNode->fctresult(0)->origin() == x);
  assert(lambdaNode->fctresult(1)->origin() == y);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Gamma", TestGamma)

static int
TestTheta()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = iostatetype::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto controlType = jlm::rvsdg::ctltype::Create(2);
  auto functionType = FunctionType::Create(
      { controlType, valueType, ioStateType },
      { controlType, valueType, ioStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode =
      lambda::node::create(rvsdg.root(), functionType, "test", linkage::external_linkage);

  auto c = lambdaNode->fctargument(0);
  auto x = lambdaNode->fctargument(1);
  auto l = lambdaNode->fctargument(2);

  auto thetaNode1 = jlm::rvsdg::ThetaNode::create(lambdaNode->subregion());
  auto thetaOutput1 = thetaNode1->add_loopvar(c);
  auto thetaOutput2 = thetaNode1->add_loopvar(x);
  auto thetaOutput3 = thetaNode1->add_loopvar(l);

  auto thetaNode2 = jlm::rvsdg::ThetaNode::create(thetaNode1->subregion());
  auto thetaOutput4 = thetaNode2->add_loopvar(thetaOutput1->argument());
  thetaNode2->add_loopvar(thetaOutput2->argument());
  auto thetaOutput5 = thetaNode2->add_loopvar(thetaOutput3->argument());
  thetaNode2->set_predicate(thetaOutput4->argument());

  thetaOutput3->result()->divert_to(thetaOutput5);
  thetaNode1->set_predicate(thetaOutput1->argument());

  auto lambdaOutput = lambdaNode->finalize({ thetaOutput1, thetaOutput2, thetaOutput3 });

  GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  assert(lambdaNode->fctresult(0)->origin() == c);
  assert(lambdaNode->fctresult(1)->origin() == x);
  assert(lambdaNode->fctresult(2)->origin() == thetaOutput3);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Theta", TestTheta)

static int
TestCall()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto controlType = jlm::rvsdg::ctltype::Create(2);
  auto functionTypeTest1 = FunctionType::Create(
      { controlType, valueType, valueType, ioStateType, memoryStateType },
      { valueType, valueType, ioStateType, memoryStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  lambda::output * lambdaOutputTest1;
  {
    auto lambdaNode =
        lambda::node::create(rvsdg.root(), functionTypeTest1, "test1", linkage::external_linkage);

    auto controlArgument = lambdaNode->fctargument(0);
    auto xArgument = lambdaNode->fctargument(1);
    auto yArgument = lambdaNode->fctargument(2);
    auto ioStateArgument = lambdaNode->fctargument(3);
    auto memoryStateArgument = lambdaNode->fctargument(4);

    auto gammaNode = jlm::rvsdg::GammaNode::create(controlArgument, 2);
    auto gammaInputX = gammaNode->AddEntryVar(xArgument);
    auto gammaInputY = gammaNode->AddEntryVar(yArgument);
    auto gammaInputIOState = gammaNode->AddEntryVar(ioStateArgument);
    auto gammaInputMemoryState = gammaNode->AddEntryVar(memoryStateArgument);
    auto gammaOutputX =
        gammaNode->add_exitvar({ gammaInputY.branches[0], gammaInputY.branches[1] });
    auto gammaOutputY =
        gammaNode->add_exitvar({ gammaInputX.branches[0], gammaInputX.branches[1] });
    auto gammaOutputIOState =
        gammaNode->add_exitvar({ gammaInputIOState.branches[0], gammaInputIOState.branches[1] });
    auto gammaOutputMemoryState = gammaNode->add_exitvar(
        { gammaInputMemoryState.branches[0], gammaInputMemoryState.branches[1] });

    lambdaOutputTest1 = lambdaNode->finalize(
        { gammaOutputX, gammaOutputY, gammaOutputIOState, gammaOutputMemoryState });
  }

  lambda::output * lambdaOutputTest2;
  {
    auto functionType = FunctionType::Create(
        { valueType, valueType, ioStateType, memoryStateType },
        { valueType, valueType, ioStateType, memoryStateType });

    auto lambdaNode =
        lambda::node::create(rvsdg.root(), functionType, "test2", linkage::external_linkage);
    auto xArgument = lambdaNode->fctargument(0);
    auto yArgument = lambdaNode->fctargument(1);
    auto ioStateArgument = lambdaNode->fctargument(2);
    auto memoryStateArgument = lambdaNode->fctargument(3);
    auto lambdaArgumentTest1 = lambdaNode->AddContextVar(lambdaOutputTest1).inner;

    auto controlResult = jlm::rvsdg::control_constant(lambdaNode->subregion(), 2, 0);

    auto & callNode = CallNode::CreateNode(
        lambdaArgumentTest1,
        functionTypeTest1,
        { controlResult, xArgument, yArgument, ioStateArgument, memoryStateArgument });

    lambdaOutputTest2 = lambdaNode->finalize(outputs(&callNode));
    GraphExport::Create(*lambdaOutputTest2, "test2");
  }

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  auto lambdaNode = lambdaOutputTest2->node();
  assert(lambdaNode->nfctresults() == 4);
  assert(lambdaNode->fctresult(0)->origin() == lambdaNode->fctargument(1));
  assert(lambdaNode->fctresult(1)->origin() == lambdaNode->fctargument(0));
  assert(lambdaNode->fctresult(2)->origin() == lambdaNode->fctargument(2));
  assert(lambdaNode->fctresult(3)->origin() == lambdaNode->fctargument(3));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/InvariantValueRedirectionTests-Call", TestCall)

static int
TestCallWithMemoryStateNodes()
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::tests::valuetype::Create();
  auto controlType = jlm::rvsdg::ctltype::Create(2);
  auto functionTypeTest1 = FunctionType::Create(
      { controlType, valueType, ioStateType, memoryStateType },
      { valueType, ioStateType, memoryStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  lambda::output * lambdaOutputTest1;
  {
    auto lambdaNode =
        lambda::node::create(rvsdg.root(), functionTypeTest1, "test1", linkage::external_linkage);

    auto controlArgument = lambdaNode->fctargument(0);
    auto xArgument = lambdaNode->fctargument(1);
    auto ioStateArgument = lambdaNode->fctargument(2);
    auto memoryStateArgument = lambdaNode->fctargument(3);

    auto lambdaEntrySplitResults =
        LambdaEntryMemoryStateSplitOperation::Create(*memoryStateArgument, 2);

    auto gammaNode = jlm::rvsdg::GammaNode::create(controlArgument, 2);

    auto gammaInputX = gammaNode->AddEntryVar(xArgument);
    auto gammaInputMemoryState1 = gammaNode->AddEntryVar(lambdaEntrySplitResults[0]);
    auto gammaInputMemoryState2 = gammaNode->AddEntryVar(lambdaEntrySplitResults[1]);

    auto gammaOutputX = gammaNode->add_exitvar(gammaInputX.branches);
    auto gammaOutputMemoryState1 = gammaNode->add_exitvar(gammaInputMemoryState2.branches);
    auto gammaOutputMemoryState2 = gammaNode->add_exitvar(gammaInputMemoryState1.branches);

    auto & lambdaExitMergeResult = LambdaExitMemoryStateMergeOperation::Create(
        *lambdaNode->subregion(),
        { gammaOutputMemoryState1, gammaOutputMemoryState2 });

    lambdaOutputTest1 =
        lambdaNode->finalize({ gammaOutputX, ioStateArgument, &lambdaExitMergeResult });
  }

  lambda::output * lambdaOutputTest2;
  {
    auto functionType = FunctionType::Create(
        { valueType, ioStateType, memoryStateType },
        { valueType, ioStateType, memoryStateType });

    auto lambdaNode =
        lambda::node::create(rvsdg.root(), functionType, "test2", linkage::external_linkage);
    auto xArgument = lambdaNode->fctargument(0);
    auto ioStateArgument = lambdaNode->fctargument(1);
    auto memoryStateArgument = lambdaNode->fctargument(2);
    auto lambdaArgumentTest1 = lambdaNode->AddContextVar(lambdaOutputTest1).inner;

    auto lambdaEntrySplitResults =
        LambdaEntryMemoryStateSplitOperation::Create(*memoryStateArgument, 2);

    auto & callEntryMergeResult = CallEntryMemoryStateMergeOperation::Create(
        *lambdaNode->subregion(),
        lambdaEntrySplitResults);

    auto controlResult = jlm::rvsdg::control_constant(lambdaNode->subregion(), 2, 0);

    auto & callNode = CallNode::CreateNode(
        lambdaArgumentTest1,
        functionTypeTest1,
        { controlResult, xArgument, ioStateArgument, &callEntryMergeResult });

    auto callExitSplitResults =
        CallExitMemoryStateSplitOperation::Create(*callNode.GetMemoryStateOutput(), 2);

    auto & lambdaExitMergeResult =
        LambdaExitMemoryStateMergeOperation::Create(*lambdaNode->subregion(), callExitSplitResults);

    lambdaOutputTest2 = lambdaNode->finalize(
        { callNode.output(0), callNode.GetIoStateOutput(), &lambdaExitMergeResult });
    GraphExport::Create(*lambdaOutputTest2, "test2");
  }

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  auto lambdaNode = lambdaOutputTest2->node();
  assert(lambdaNode->nfctresults() == 3);
  assert(lambdaNode->fctresult(0)->origin() == lambdaNode->fctargument(0));
  assert(lambdaNode->fctresult(1)->origin() == lambdaNode->fctargument(1));

  auto lambdaEntrySplit = lambda::node::GetMemoryStateEntrySplit(*lambdaNode);
  auto lambdaExitMerge = lambda::node::GetMemoryStateExitMerge(*lambdaNode);

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

  assert(lambdaNode.nfctresults() == 3);
  assert(lambdaNode.nfctresults() == callNode.NumResults());
  assert(lambdaNode.fctresult(0)->origin() == callNode.Result(0));
  assert(lambdaNode.fctresult(1)->origin() == callNode.Result(1));
  assert(lambdaNode.fctresult(2)->origin() == callNode.Result(2));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/InvariantValueRedirectionTests-LambdaCallArgumentMismatch",
    TestLambdaCallArgumentMismatch)
