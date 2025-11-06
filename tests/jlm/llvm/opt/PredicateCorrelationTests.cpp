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

struct MatchConstantCorrelationTest
{
  jlm::rvsdg::GammaNode & gammaNode;
  jlm::rvsdg::ThetaNode & thetaNode;
  jlm::rvsdg::Node & matchNode;
};

static MatchConstantCorrelationTest
setupMatchConstantCorrelationTest(
    jlm::rvsdg::Graph & rvsdg,
    const std::pair<uint64_t, uint64_t> & gammaSubregionAlternatives)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto predicate = TestOperation::create(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode = GammaNode::create(predicate, 2);

  auto constant0 =
      create_bitconstant(gammaNode->subregion(0), 64, gammaSubregionAlternatives.first);
  auto constant1 =
      create_bitconstant(gammaNode->subregion(1), 64, gammaSubregionAlternatives.second);

  auto exitVar = gammaNode->AddExitVar({ constant0, constant1 });

  auto & matchNode = MatchOperation::CreateNode(*exitVar.output, { { 1, 1 } }, 0, 2);

  thetaNode->predicate()->divert_to(matchNode.output(0));

  return { *gammaNode, *thetaNode, matchNode };
}

struct MatchCorrelationTest
{
  jlm::rvsdg::GammaNode & gammaNode;
  jlm::rvsdg::ThetaNode & thetaNode;
  jlm::rvsdg::Node & matchNode;
};

static MatchCorrelationTest
setupMatchCorrelationTest(jlm::rvsdg::Graph & rvsdg)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto constantNode = TestOperation::create(thetaNode->subregion(), {}, { bitType32 });
  auto & matchNode = MatchOperation::CreateNode(*constantNode->output(0), { { 1, 1 } }, 0, 2);

  auto gammaNode = GammaNode::create(matchNode.output(0), 2);

  thetaNode->predicate()->divert_to(matchNode.output(0));

  return { *gammaNode, *thetaNode, matchNode };
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

  auto controlConstant0 = &ControlConstantOperation::create(*gammaNode->subregion(0), 2, 0);
  auto controlConstant1 = &ControlConstantOperation::create(*gammaNode->subregion(1), 2, 1);

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

  const std::vector<std::pair<uint64_t, uint64_t>> gammaSubregionAlternatives = { { 0, 1 },
                                                                                  { 1, 0 } };
  for (auto alternatives : gammaSubregionAlternatives)
  {
    // Arrange
    auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto & rvsdg = rvsdgModule->Rvsdg();

    auto [gammaNode, thetaNode, matchNode] = setupMatchConstantCorrelationTest(rvsdg, alternatives);

    view(rvsdg, stdout);

    // Act
    const auto correlationOpt = computeThetaGammaPredicateCorrelation(thetaNode);

    // Assert
    assert(correlationOpt.value() != nullptr);
    assert(correlationOpt.value()->type() == CorrelationType::MatchConstantCorrelation);
    assert(&correlationOpt.value()->thetaNode() == &thetaNode);
    assert(&correlationOpt.value()->gammaNode() == &gammaNode);

    const auto correlationData =
        std::get<ThetaGammaPredicateCorrelation::MatchConstantCorrelationData>(
            correlationOpt.value()->data());
    assert(correlationData.matchNode == &matchNode);
    assert(correlationData.alternatives.size() == 2);
    assert(correlationData.alternatives[0] == alternatives.first);
    assert(correlationData.alternatives[1] == alternatives.second);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/PredicateCorrelationTests-testMatchConstantCorrelationDetection",
    testMatchConstantCorrelationDetection)

static void
testMatchConstantCorrelation_Success()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto [gammaNode, thetaNode, _] = setupMatchConstantCorrelationTest(rvsdg, { 0, 1 });
  auto gammaPredicate = gammaNode.predicate()->origin();
  view(rvsdg, stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  PredicateCorrelation predicateCorrelation;
  predicateCorrelation.Run(*rvsdgModule, statisticsCollector);

  thetaNode.subregion()->prune(true);

  view(rvsdg, stdout);

  // Assert
  assert(thetaNode.subregion()->numNodes() == 1);
  assert(thetaNode.predicate()->origin() == gammaPredicate);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/PredicateCorrelationTests-testMatchConstantCorrelation_Success",
    testMatchConstantCorrelation_Success)

static void
testMatchConstantCorrelation_Failure()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto gammaSubregionAlternatives = std::make_pair(1, 0);
  auto [gammaNode, thetaNode, _] =
      setupMatchConstantCorrelationTest(rvsdg, gammaSubregionAlternatives);
  auto gammaPredicate = gammaNode.predicate()->origin();
  view(rvsdg, stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  PredicateCorrelation predicateCorrelation;
  predicateCorrelation.Run(*rvsdgModule, statisticsCollector);

  thetaNode.subregion()->prune(true);

  view(rvsdg, stdout);

  // Assert
  // The theta node predicate is not redirected as the gamma subregion alternatives do not lead to
  // the same control behavior as the match node that is currently connected to the theta node
  // predicate. It would be necessary to create a new match node for this instead of just reusing
  // the gamma node's control predicate.
  assert(thetaNode.predicate()->origin() != gammaPredicate);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/PredicateCorrelationTests-testMatchConstantCorrelation_Failure",
    testMatchConstantCorrelation_Failure)

static void
testMatchCorrelationDetection()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto [gammaNode, thetaNode, matchNode] = setupMatchCorrelationTest(rvsdg);

  view(rvsdg, stdout);

  // Act
  const auto correlationOpt = computeThetaGammaPredicateCorrelation(thetaNode);

  // Assert
  assert(correlationOpt.value() != nullptr);
  assert(correlationOpt.value()->type() == CorrelationType::MatchCorrelation);
  assert(&correlationOpt.value()->thetaNode() == &thetaNode);
  assert(&correlationOpt.value()->gammaNode() == &gammaNode);

  const auto correlationData = std::get<ThetaGammaPredicateCorrelation::MatchCorrelationData>(
      correlationOpt.value()->data());
  assert(correlationData.matchNode == &matchNode);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/PredicateCorrelationTests-testMatchCorrelationDetection",
    testMatchCorrelationDetection)
