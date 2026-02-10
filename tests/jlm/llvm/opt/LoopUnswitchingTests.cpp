/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/LoopUnswitching.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Statistics.hpp>

TEST(LoopUnswitchingTests, Test1)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::rvsdg::TestType::createValueType();

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, valueType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, valueType, "z");

  auto thetaNode = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto loopVarX = thetaNode->AddLoopVar(x);
  auto loopVarY = thetaNode->AddLoopVar(y);
  thetaNode->AddLoopVar(z);

  auto a = TestOperation::createNode(
               thetaNode->subregion(),
               { loopVarX.pre, loopVarY.pre },
               { jlm::rvsdg::BitType::Create(1) })
               ->output(0);
  auto & predicateNode = MatchOperation::CreateNode(*a, { { 1, 0 } }, 1, 2);

  auto gamma = GammaNode::create(predicateNode.output(0), 2);

  auto entryVarX = gamma->AddEntryVar(loopVarX.pre);
  auto entryVarY = gamma->AddEntryVar(loopVarY.pre);

  auto b = TestOperation::createNode(
               gamma->subregion(0),
               { entryVarX.branchArgument[0], entryVarY.branchArgument[0] },
               { valueType })
               ->output(0);
  auto c = TestOperation::createNode(
               gamma->subregion(1),
               { entryVarX.branchArgument[1], entryVarY.branchArgument[1] },
               { valueType })
               ->output(0);

  auto exitVarY = gamma->AddExitVar({ b, c });

  loopVarY.post->divert_to(exitVarY.output);

  thetaNode->set_predicate(predicateNode.output(0));

  auto & ex1 = jlm::rvsdg::GraphExport::Create(*thetaNode->output(0), "x");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*thetaNode->output(1), "y");
  auto & ex3 = jlm::rvsdg::GraphExport::Create(*thetaNode->output(2), "z");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  LoopUnswitching::CreateAndRun(rvsdgModule, statisticsCollector);

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_NE(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex1.origin()), nullptr);
  EXPECT_NE(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex2.origin()), nullptr);
  EXPECT_NE(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex3.origin()), nullptr);
}

TEST(LoopUnswitchingTests, Test2)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::rvsdg::TestType::createValueType();

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");

  auto thetaNode = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto loopVarX = thetaNode->AddLoopVar(x);

  auto n1 = TestOperation::createNode(
                thetaNode->subregion(),
                { loopVarX.pre },
                { jlm::rvsdg::BitType::Create(1) })
                ->output(0);
  auto n2 =
      TestOperation::createNode(thetaNode->subregion(), { loopVarX.pre }, { valueType })->output(0);
  auto & predicateNode = MatchOperation::CreateNode(*n1, { { 1, 0 } }, 1, 2);

  auto gammaNode = GammaNode::create(predicateNode.output(0), 2);

  auto ev1 = gammaNode->AddEntryVar(n1);
  auto ev2 = gammaNode->AddEntryVar(loopVarX.pre);
  auto ev3 = gammaNode->AddEntryVar(n2);

  gammaNode->AddExitVar(ev1.branchArgument);
  gammaNode->AddExitVar(ev2.branchArgument);
  gammaNode->AddExitVar(ev3.branchArgument);

  loopVarX.post->divert_to(gammaNode->output(1));

  thetaNode->set_predicate(predicateNode.output(0));

  auto & ex = jlm::rvsdg::GraphExport::Create(*thetaNode->output(0), "x");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  LoopUnswitching::CreateAndRun(rvsdgModule, statisticsCollector);

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_NE(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex.origin()), nullptr);
}
