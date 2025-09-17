/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/LoopUnswitching.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
Test1()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, valueType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, valueType, "z");

  auto thetaNode = ThetaNode::create(&graph.GetRootRegion());

  auto loopVarX = thetaNode->AddLoopVar(x);
  auto loopVarY = thetaNode->AddLoopVar(y);
  thetaNode->AddLoopVar(z);

  auto a = jlm::tests::TestOperation::create(
               thetaNode->subregion(),
               { loopVarX.pre, loopVarY.pre },
               { BitType::Create(1) })
               ->output(0);
  auto predicate = match(1, { { 1, 1 } }, 0, 2, a);

  auto gammaNode = GammaNode::create(predicate, 2);

  auto entryVarX = gammaNode->AddEntryVar(loopVarX.pre);
  auto entryVarY = gammaNode->AddEntryVar(loopVarY.pre);

  // Gamma subregion 0
  auto b = jlm::tests::TestOperation::create(
               gammaNode->subregion(0),
               { entryVarX.branchArgument[0], entryVarY.branchArgument[0] },
               { valueType })
               ->output(0);

  auto & ctlConstantNode0 = jlm::rvsdg::CreateOpNode<ctlconstant_op>(
      *gammaNode->subregion(0),
      ControlValueRepresentation(0, 2));

  // Gamma subregion 1
  auto c = jlm::tests::TestOperation::create(
               gammaNode->subregion(1),
               { entryVarX.branchArgument[1], entryVarY.branchArgument[1] },
               { valueType })
               ->output(0);

  auto & ctlConstantNode1 = jlm::rvsdg::CreateOpNode<ctlconstant_op>(
      *gammaNode->subregion(1),
      ControlValueRepresentation(1, 2));

  auto exitVarY = gammaNode->AddExitVar({ b, c });
  auto exitVarCtl =
      gammaNode->AddExitVar({ ctlConstantNode0.output(0), ctlConstantNode1.output(0) });

  loopVarY.post->divert_to(exitVarY.output);

  thetaNode->set_predicate(exitVarCtl.output);

  auto & ex1 = GraphExport::Create(*thetaNode->output(0), "x");
  auto & ex2 = GraphExport::Create(*thetaNode->output(1), "y");
  auto & ex3 = GraphExport::Create(*thetaNode->output(2), "z");

  view(graph, stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  LoopUnswitching::CreateAndRun(rvsdgModule, statisticsCollector);

  view(graph, stdout);

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
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");

  auto thetaNode = ThetaNode::create(&graph.GetRootRegion());

  auto loopVarX = thetaNode->AddLoopVar(x);

  auto n1 = jlm::tests::TestOperation::create(
                thetaNode->subregion(),
                { loopVarX.pre },
                { BitType::Create(1) })
                ->output(0);
  auto n2 =
      jlm::tests::TestOperation::create(thetaNode->subregion(), { loopVarX.pre }, { valueType })
          ->output(0);
  auto predicate = match(1, { { 1, 1 } }, 0, 2, n1);

  auto gammaNode = GammaNode::create(predicate, 2);

  auto ev1 = gammaNode->AddEntryVar(n1);
  auto ev2 = gammaNode->AddEntryVar(loopVarX.pre);
  auto ev3 = gammaNode->AddEntryVar(n2);

  auto & ctlConstantNode0 = jlm::rvsdg::CreateOpNode<ctlconstant_op>(
      *gammaNode->subregion(0),
      ControlValueRepresentation(0, 2));

  auto & ctlConstantNode1 = jlm::rvsdg::CreateOpNode<ctlconstant_op>(
      *gammaNode->subregion(1),
      ControlValueRepresentation(1, 2));

  gammaNode->AddExitVar(ev1.branchArgument);
  gammaNode->AddExitVar(ev2.branchArgument);
  gammaNode->AddExitVar(ev3.branchArgument);
  auto exitVarCtl =
      gammaNode->AddExitVar({ ctlConstantNode0.output(0), ctlConstantNode1.output(0) });

  loopVarX.post->divert_to(gammaNode->output(1));

  thetaNode->set_predicate(exitVarCtl.output);

  auto & ex = GraphExport::Create(*thetaNode->output(0), "x");

  view(graph, stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  LoopUnswitching::CreateAndRun(rvsdgModule, statisticsCollector);

  view(graph, stdout);

  // Assert
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex.origin()));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoopUnswitchingTests-Test2", Test2)