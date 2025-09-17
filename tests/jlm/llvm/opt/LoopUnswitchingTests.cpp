/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/LoopUnswitching.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
Test1()
{
  using namespace jlm::llvm;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, valueType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, valueType, "z");

  auto thetaNode = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto loopVarX = thetaNode->AddLoopVar(x);
  auto loopVarY = thetaNode->AddLoopVar(y);
  thetaNode->AddLoopVar(z);

  auto a = jlm::tests::TestOperation::create(
               thetaNode->subregion(),
               { loopVarX.pre, loopVarY.pre },
               { jlm::rvsdg::BitType::Create(1) })
               ->output(0);
  auto predicate = jlm::rvsdg::match(1, { { 1, 0 } }, 1, 2, a);

  auto gamma = jlm::rvsdg::GammaNode::create(predicate, 2);

  auto entryVarX = gamma->AddEntryVar(loopVarX.pre);
  auto entryVarY = gamma->AddEntryVar(loopVarY.pre);

  auto b = jlm::tests::TestOperation::create(
               gamma->subregion(0),
               { entryVarX.branchArgument[0], entryVarY.branchArgument[0] },
               { valueType })
               ->output(0);
  auto c = jlm::tests::TestOperation::create(
               gamma->subregion(1),
               { entryVarX.branchArgument[1], entryVarY.branchArgument[1] },
               { valueType })
               ->output(0);

  auto exitVarY = gamma->AddExitVar({ b, c });

  loopVarY.post->divert_to(exitVarY.output);

  thetaNode->set_predicate(predicate);

  auto & ex1 = jlm::rvsdg::GraphExport::Create(*thetaNode->output(0), "x");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*thetaNode->output(1), "y");
  auto & ex3 = jlm::rvsdg::GraphExport::Create(*thetaNode->output(2), "z");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  LoopUnswitching::CreateAndRun(rvsdgModule, statisticsCollector);

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Assert
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex1.origin()));
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex2.origin()));
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex3.origin()));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoopUnswitchingTests-Test1", Test1)

static void
Test2()
{
  using namespace jlm::llvm;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");

  auto thetaNode = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto loopVarX = thetaNode->AddLoopVar(x);

  auto n1 = jlm::tests::TestOperation::create(
                thetaNode->subregion(),
                { loopVarX.pre },
                { jlm::rvsdg::BitType::Create(1) })
                ->output(0);
  auto n2 =
      jlm::tests::TestOperation::create(thetaNode->subregion(), { loopVarX.pre }, { valueType })
          ->output(0);
  auto predicate = jlm::rvsdg::match(1, { { 1, 0 } }, 1, 2, n1);

  auto gammaNode = jlm::rvsdg::GammaNode::create(predicate, 2);

  auto ev1 = gammaNode->AddEntryVar(n1);
  auto ev2 = gammaNode->AddEntryVar(loopVarX.pre);
  auto ev3 = gammaNode->AddEntryVar(n2);

  gammaNode->AddExitVar(ev1.branchArgument);
  gammaNode->AddExitVar(ev2.branchArgument);
  gammaNode->AddExitVar(ev3.branchArgument);

  loopVarX.post->divert_to(gammaNode->output(1));

  thetaNode->set_predicate(predicate);

  auto & ex = jlm::rvsdg::GraphExport::Create(*thetaNode->output(0), "x");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  LoopUnswitching::CreateAndRun(rvsdgModule, statisticsCollector);

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Assert
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex.origin()));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoopUnswitchingTests-Test2", Test2)

static void
Test3()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & zeroNode = IntegerConstantOperation::Create(rvsdg.GetRootRegion(), 32, 0);

  auto thetaNode = jlm::rvsdg::ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVar = thetaNode->AddLoopVar(zeroNode.output(0));

  auto & oneNode = IntegerConstantOperation::Create(*thetaNode->subregion(), 32, 1);
  auto & fiveNode = IntegerConstantOperation::Create(*thetaNode->subregion(), 32, 5);

  auto & addNode = CreateOpNode<IntegerAddOperation>({ loopVar.pre, oneNode.output(0) }, 32);
  auto & ultNode = CreateOpNode<IntegerUltOperation>({ addNode.output(0), fiveNode.output(0) }, 32);

  auto matchResult = MatchOperation::Create(*ultNode.output(0), { { 1, 1 } }, 0, 2);

  auto gammaNode = GammaNode::create(matchResult, 2);
  auto entryVar1 = gammaNode->AddEntryVar(addNode.output(0));
  auto entryVar2 = gammaNode->AddEntryVar(loopVar.pre);

  auto controlZero = control_constant(gammaNode->subregion(0), 2, 0);
  auto controlOne = control_constant(gammaNode->subregion(1), 2, 1);

  auto exitVarCtl = gammaNode->AddExitVar({ controlZero, controlOne });
  auto exitVarIV =
      gammaNode->AddExitVar({ entryVar2.branchArgument[0], entryVar1.branchArgument[1] });

  loopVar.post->divert_to(exitVarIV.output);
  thetaNode->set_predicate(exitVarCtl.output);

  GraphExport::Create(*thetaNode->output(0), "x");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  LoopUnswitching::CreateAndRun(rvsdgModule, statisticsCollector);

  view(&rvsdg.GetRootRegion(), stdout);

  // Assert
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoopUnswitchingTests-Test3", Test3)
