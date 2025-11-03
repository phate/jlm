/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/opt/PredicateCorrelation.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static std::unique_ptr<jlm::llvm::RvsdgModule>
setupMatchConstantCorrelationTest()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto predicate = TestOperation::create(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode = GammaNode::create(predicate, 2);

  auto constant0 = create_bitconstant(gammaNode->subregion(0), 2, 0);
  auto constant1 = create_bitconstant(gammaNode->subregion(1), 2, 1);

  auto exitVar = gammaNode->AddExitVar({ constant0, constant1 });

  auto & matchNode = MatchOperation::CreateNode(*exitVar.output, { { 1, 1 } }, 0, 2);

  thetaNode->predicate()->divert_to(matchNode.output(0));

  return rvsdgModule;
}

static void
testControlConstantCorrelation()
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

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/PredicateCorrelationTests-testControlConstantCorrelation",
    testControlConstantCorrelation)

static void
testMatchConstantCorrelationDetection()
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

  auto predicate = TestOperation::create(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode = GammaNode::create(predicate, 2);

  auto constant0 = create_bitconstant(gammaNode->subregion(0), 2, 0);
  auto constant1 = create_bitconstant(gammaNode->subregion(1), 2, 1);

  auto exitVar = gammaNode->AddExitVar({ constant0, constant1 });

  auto & matchNode = MatchOperation::CreateNode(*exitVar.output, { { 1, 1 } }, 0, 2);

  thetaNode->predicate()->divert_to(matchNode.output(0));

  view(rvsdg, stdout);

  // Act
  const auto correlationOpt = computeThetaGammaPredicateCorrelation(*thetaNode);

  // Assert
  assert(correlationOpt.value() != nullptr);
  assert(correlationOpt.value()->type() == CorrelationType::MatchConstantCorrelation);
  assert(&correlationOpt.value()->thetaNode() == thetaNode);
  assert(&correlationOpt.value()->gammaNode() == gammaNode);

  const auto correlationData =
      std::get<ThetaGammaPredicateCorrelation::MatchConstantCorrelationData>(
          correlationOpt.value()->data());
  assert(correlationData.matchNode == &matchNode);
  assert(correlationData.alternatives.size() == 2);
  assert(correlationData.alternatives[0] == 0);
  assert(correlationData.alternatives[1] == 1);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/PredicateCorrelationTests-testMatchConstantCorrelationDetection",
    testMatchConstantCorrelationDetection)

static void
testMatchConstantCorrelation()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto rvsdgModule = setupMatchConstantCorrelationTest();
  auto & rvsdg = rvsdgModule->Rvsdg();

#if 0
  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto predicate = TestOperation::create(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode = GammaNode::create(predicate, 2);

  auto constant0 = create_bitconstant(gammaNode->subregion(0), 2, 0);
  auto constant1 = create_bitconstant(gammaNode->subregion(1), 2, 1);

  auto exitVar = gammaNode->AddExitVar({ constant0, constant1 });

  auto & matchNode = MatchOperation::CreateNode(*exitVar.output, { { 1, 1 } }, 0, 2);

  thetaNode->predicate()->divert_to(matchNode.output(0));
#endif
  view(rvsdg, stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  PredicateCorrelation predicateCorrelation;
  predicateCorrelation.Run(*rvsdgModule, statisticsCollector);

  thetaNode->subregion()->prune(true);

  view(rvsdg, stdout);

  // Assert
  assert(thetaNode->subregion()->numNodes() == 1);
  assert(thetaNode->predicate()->origin() == predicate);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/PredicateCorrelationTests-testMatchConstantCorrelation",
    testMatchConstantCorrelation)
