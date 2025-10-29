/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/opt/PredicateCorrelation.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static void
testThetaGamma()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto dummy = TestOperation::create(thetaNode->subregion(), {}, { bitType32 })->output(0);
  auto predicate = MatchOperation::Create(*dummy, { { 1, 1 } }, 0, 2);

  auto gammaNode = GammaNode::create(predicate, 2);

  auto controlConstant0 = control_constant(gammaNode->subregion(0), 2, 0);
  auto controlConstant1 = control_constant(gammaNode->subregion(1), 2, 1);

  auto controlExitVar = gammaNode->AddExitVar({ controlConstant0, controlConstant1 });

  thetaNode->predicate()->divert_to(controlExitVar.output);

  view(rvsdg, stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  PredicateCorrelation predicateCorrelation;
  predicateCorrelation.Run(*rvsdgModule, statisticsCollector);

  thetaNode->subregion()->prune(true);

  view(rvsdg, stdout);

  // Assert
  assert(thetaNode->subregion()->numNodes() == 2);
  assert(thetaNode->predicate()->origin() == predicate);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/PredicateCorrelationTests-testThetaGamma", testThetaGamma)
