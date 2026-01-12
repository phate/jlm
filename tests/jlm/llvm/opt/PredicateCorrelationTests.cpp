/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/opt/PredicateCorrelation.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

struct ControlConstantCorrelationTest
{
  jlm::rvsdg::GammaNode & gammaNode;
  jlm::rvsdg::ThetaNode & thetaNode;
  jlm::rvsdg::Node & matchNode;
  std::vector<jlm::rvsdg::Node *> controlConstants{};
};

static ControlConstantCorrelationTest
setupControlConstantCorrelationTest(
    jlm::rvsdg::Graph & rvsdg,
    const std::pair<uint64_t, uint64_t> & gammaSubregionControlConstants)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto dummy = TestOperation::createNode(thetaNode->subregion(), {}, { bitType32 })->output(0);
  auto predicate = MatchOperation::Create(*dummy, { { 1, 1 } }, 0, 2);

  auto gammaNode = GammaNode::create(predicate, 2);

  auto controlConstant0 = &ControlConstantOperation::create(
      *gammaNode->subregion(0),
      2,
      gammaSubregionControlConstants.first);
  auto controlConstant1 = &ControlConstantOperation::create(
      *gammaNode->subregion(1),
      2,
      gammaSubregionControlConstants.second);

  auto controlExitVar = gammaNode->AddExitVar({ controlConstant0, controlConstant1 });

  thetaNode->predicate()->divert_to(controlExitVar.output);

  return { *gammaNode,
           *thetaNode,
           *TryGetOwnerNode<Node>(*predicate),
           { TryGetOwnerNode<Node>(*controlConstant0), TryGetOwnerNode<Node>(*controlConstant1) } };
}

struct MatchConstantCorrelationTest
{
  jlm::rvsdg::GammaNode & gammaNode;
  jlm::rvsdg::ThetaNode & thetaNode;
  jlm::rvsdg::Node & matchNode;
};

static MatchConstantCorrelationTest
setupMatchConstantCorrelationTest(
    jlm::rvsdg::Graph & rvsdg,
    const std::pair<int64_t, int64_t> & gammaSubregionAlternatives)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto predicate =
      TestOperation::createNode(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode = GammaNode::create(predicate, 2);

  auto constant0 = &BitConstantOperation::create(
      *gammaNode->subregion(0),
      { 64, gammaSubregionAlternatives.first });
  auto constant1 = &BitConstantOperation::create(
      *gammaNode->subregion(1),
      { 64, gammaSubregionAlternatives.second });

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
setupThetaGammaMatchCorrelationTest(jlm::rvsdg::Graph & rvsdg)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto constantNode = TestOperation::createNode(thetaNode->subregion(), {}, { bitType32 });
  auto & matchNode = MatchOperation::CreateNode(*constantNode->output(0), { { 1, 1 } }, 0, 2);

  auto gammaNode = GammaNode::create(matchNode.output(0), 2);

  thetaNode->predicate()->divert_to(matchNode.output(0));

  return { *gammaNode, *thetaNode, matchNode };
}

TEST(PredicateCorrelationTests, testControlConstantCorrelation)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto [gammaNode, thetaNode, matchNode, controlConstants] =
      setupControlConstantCorrelationTest(rvsdg, { 0, 1 });

  view(rvsdg, stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  PredicateCorrelation predicateCorrelation;
  predicateCorrelation.Run(*rvsdgModule, statisticsCollector);

  thetaNode.subregion()->prune(true);

  view(rvsdg, stdout);

  // Assert
  EXPECT_EQ(thetaNode.subregion()->numNodes(), 2u);
  EXPECT_EQ(thetaNode.predicate()->origin(), matchNode.output(0));
}

TEST(PredicateCorrelationTests, testMatchConstantCorrelationDetection)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  const std::vector<std::pair<uint64_t, uint64_t>> gammaSubregionAlternatives = { { 0, 1 },
                                                                                  { 1, 0 } };
  for (auto alternatives : gammaSubregionAlternatives)
  {
    // Arrange
    auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto & rvsdg = rvsdgModule->Rvsdg();

    auto [gammaNode, thetaNode, matchNode] = setupMatchConstantCorrelationTest(rvsdg, alternatives);

    view(rvsdg, stdout);

    // Act
    const auto correlationOpt = computeThetaGammaPredicateCorrelation(thetaNode);

    // Assert
    EXPECT_NE(correlationOpt.value(), nullptr);
    EXPECT_EQ(correlationOpt.value()->type(), CorrelationType::MatchConstantCorrelation);
    EXPECT_EQ(&correlationOpt.value()->thetaNode(), &thetaNode);
    EXPECT_EQ(&correlationOpt.value()->gammaNode(), &gammaNode);

    const auto correlationData =
        std::get<ThetaGammaPredicateCorrelation::MatchConstantCorrelationData>(
            correlationOpt.value()->data());
    EXPECT_EQ(correlationData.matchNode, &matchNode);
    EXPECT_EQ(correlationData.alternatives.size(), 2u);
    EXPECT_EQ(correlationData.alternatives[0], alternatives.first);
    EXPECT_EQ(correlationData.alternatives[1], alternatives.second);
  }
}

TEST(PredicateCorrelationTests, testMatchConstantCorrelation_Success)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
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
  EXPECT_EQ(thetaNode.subregion()->numNodes(), 1u);
  EXPECT_EQ(thetaNode.predicate()->origin(), gammaPredicate);
}

TEST(PredicateCorrelationTests, testMatchConstantCorrelation_Failure)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
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
  EXPECT_NE(thetaNode.predicate()->origin(), gammaPredicate);
}

TEST(PredicateCorrelationTests, testThetaGammaMatchCorrelationDetection)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto [gammaNode, thetaNode, matchNode] = setupThetaGammaMatchCorrelationTest(rvsdg);

  view(rvsdg, stdout);

  // Act
  const auto correlationOpt = computeThetaGammaPredicateCorrelation(thetaNode);

  // Assert
  EXPECT_NE(correlationOpt.value(), nullptr);
  EXPECT_EQ(correlationOpt.value()->type(), CorrelationType::MatchCorrelation);
  EXPECT_EQ(&correlationOpt.value()->thetaNode(), &thetaNode);
  EXPECT_EQ(&correlationOpt.value()->gammaNode(), &gammaNode);

  const auto correlationData = std::get<ThetaGammaPredicateCorrelation::MatchCorrelationData>(
      correlationOpt.value()->data());
  EXPECT_EQ(correlationData.matchNode, &matchNode);
}

TEST(PredicateCorrelationTests, testThetaGammaCorrelationFixPoint)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  // Arrange first gamma node
  auto predicate =
      TestOperation::createNode(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode1 = GammaNode::create(predicate, 2);

  auto constant0 = &BitConstantOperation::create(*gammaNode1->subregion(0), { 64, 0 });
  auto constant1 = &BitConstantOperation::create(*gammaNode1->subregion(1), { 64, 1 });

  auto exitVar = gammaNode1->AddExitVar({ constant0, constant1 });
  auto & matchNode = MatchOperation::CreateNode(*exitVar.output, { { 1, 1 } }, 0, 2);

  // Arrange second gamma node
  auto gammaNode2 = GammaNode::create(matchNode.output(0), 2);

  auto controlConstant0 = &ControlConstantOperation::create(*gammaNode2->subregion(0), 2, 0);
  auto controlConstant1 = &ControlConstantOperation::create(*gammaNode2->subregion(1), 2, 1);

  auto controlExitVar = gammaNode2->AddExitVar({ controlConstant0, controlConstant1 });

  thetaNode->predicate()->divert_to(controlExitVar.output);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  PredicateCorrelation predicateCorrelation;
  predicateCorrelation.Run(*rvsdgModule, statisticsCollector);

  thetaNode->subregion()->prune(true);

  view(rvsdg, stdout);

  // Assert
  EXPECT_EQ(thetaNode->subregion()->numNodes(), 1u);
  EXPECT_EQ(thetaNode->predicate()->origin(), predicate);
}

TEST(PredicateCorrelationTests, testDetermineGammaSubregionRoles_ControlConstantCorrelation)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  {
    // Arrange
    auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto & rvsdg = rvsdgModule->Rvsdg();

    constexpr std::pair<uint64_t, uint64_t> controlAlternatives = { 0, 1 };
    auto [gammaNode, thetaNode, matchNode, controlConstants] =
        setupControlConstantCorrelationTest(rvsdg, controlAlternatives);

    const auto correlation = ThetaGammaPredicateCorrelation::CreateControlConstantCorrelation(
        thetaNode,
        gammaNode,
        { controlAlternatives.first, controlAlternatives.second });

    // Act
    const auto gammaSubregionRoles = determineGammaSubregionRoles(*correlation);

    // Assert
    EXPECT_EQ(gammaSubregionRoles->exitSubregion, gammaNode.subregion(0));
    EXPECT_EQ(gammaSubregionRoles->repetitionSubregion, gammaNode.subregion(1));
  }

  {
    // Arrange
    auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto & rvsdg = rvsdgModule->Rvsdg();

    constexpr std::pair<uint64_t, uint64_t> controlAlternatives = { 1, 0 };
    auto [gammaNode, thetaNode, matchNode, controlConstants] =
        setupControlConstantCorrelationTest(rvsdg, controlAlternatives);

    const auto correlation = ThetaGammaPredicateCorrelation::CreateControlConstantCorrelation(
        thetaNode,
        gammaNode,
        { controlAlternatives.first, controlAlternatives.second });

    // Act
    const auto gammaSubregionRoles = determineGammaSubregionRoles(*correlation);

    // Assert
    EXPECT_EQ(gammaSubregionRoles->exitSubregion, gammaNode.subregion(1));
    EXPECT_EQ(gammaSubregionRoles->repetitionSubregion, gammaNode.subregion(0));
  }
}

TEST(PredicateCorrelationTests, testDetermineGammaSubregionRoles_MatchConstantCorrelation)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  {
    // Arrange
    auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto & rvsdg = rvsdgModule->Rvsdg();

    constexpr std::pair<uint64_t, uint64_t> gammaSubregionAlternatives = { 0, 1 };
    auto [gammaNode, thetaNode, matchNode] =
        setupMatchConstantCorrelationTest(rvsdg, gammaSubregionAlternatives);

    const auto correlation = ThetaGammaPredicateCorrelation::CreateMatchConstantCorrelation(
        thetaNode,
        gammaNode,
        { &matchNode, { gammaSubregionAlternatives.first, gammaSubregionAlternatives.second } });

    // Act
    const auto gammaSubregionRoles = determineGammaSubregionRoles(*correlation);

    // Assert
    EXPECT_EQ(gammaSubregionRoles->exitSubregion, gammaNode.subregion(0));
    EXPECT_EQ(gammaSubregionRoles->repetitionSubregion, gammaNode.subregion(1));
  }

  {
    // Arrange
    auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto & rvsdg = rvsdgModule->Rvsdg();

    constexpr std::pair<uint64_t, uint64_t> gammaSubregionAlternatives = { 1, 0 };
    auto [gammaNode, thetaNode, matchNode] =
        setupMatchConstantCorrelationTest(rvsdg, gammaSubregionAlternatives);

    const auto correlation = ThetaGammaPredicateCorrelation::CreateMatchConstantCorrelation(
        thetaNode,
        gammaNode,
        { &matchNode, { gammaSubregionAlternatives.first, gammaSubregionAlternatives.second } });

    // Act
    const auto gammaSubregionRoles = determineGammaSubregionRoles(*correlation);

    // Assert
    EXPECT_EQ(gammaSubregionRoles->exitSubregion, gammaNode.subregion(1));
    EXPECT_EQ(gammaSubregionRoles->repetitionSubregion, gammaNode.subregion(0));
  }
}

TEST(PredicateCorrelationTests, testDetermineGammaSubregionRoles_MatchCorrelation)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  // Arrange
  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto [gammaNode, thetaNode, matchNode] = setupThetaGammaMatchCorrelationTest(rvsdg);

  const auto correlation =
      ThetaGammaPredicateCorrelation::CreateMatchCorrelation(thetaNode, gammaNode, { &matchNode });

  // Act
  const auto gammaSubregionRoles = determineGammaSubregionRoles(*correlation);

  // Assert
  EXPECT_EQ(gammaSubregionRoles->exitSubregion, gammaNode.subregion(0));
  EXPECT_EQ(gammaSubregionRoles->repetitionSubregion, gammaNode.subregion(1));
}

TEST(PredicateCorrelationTests, testGammaGammaMatchCorrelationDetection)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType32 = BitType::Create(32);
  auto controlType = ControlType::Create(2);

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto constantNode = TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType32 });
  auto & matchNode = MatchOperation::CreateNode(*constantNode->output(0), { { 1, 1 } }, 0, 2);

  auto gammaNode1 = GammaNode::create(matchNode.output(0), 2);

  auto gammaNode2 = GammaNode::create(matchNode.output(0), 2);

  view(rvsdg, stdout);

  // Act
  const auto correlationOpt = computeGammaGammaPredicateCorrelation(*gammaNode1);

  // Assert
  EXPECT_NE(correlationOpt.value(), nullptr);
  EXPECT_EQ(correlationOpt.value()->type(), CorrelationType::MatchCorrelation);
  EXPECT_EQ(&correlationOpt.value()->gammaNode1(), gammaNode1);
  EXPECT_EQ(&correlationOpt.value()->gammaNode2(), gammaNode2);

  const auto correlationData = std::get<GammaGammaPredicateCorrelation::MatchCorrelationData>(
      correlationOpt.value()->correlationData());
  EXPECT_EQ(correlationData.matchNode, &matchNode);
}
