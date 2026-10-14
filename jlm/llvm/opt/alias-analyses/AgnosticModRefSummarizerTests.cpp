/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunAndersen(const jlm::llvm::LlvmRvsdgModule & module)
{
  using namespace jlm::llvm;

  aa::Andersen andersen;
  jlm::util::StatisticsCollector statisticsCollector;
  return andersen.Analyze(module, statisticsCollector);
}

TEST(AgnosticModRefSummarizerTests, TestStore1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::StoreTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto lambdaPtgNode = pointsToGraph.getNodeForLambda(*test.lambda);

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).getModRefNodes();
    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).getModRefNodes();
    EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

    // The memory location representing the function itself should not be included
    EXPECT_EQ(lambdaEntryNodes.count(lambdaPtgNode), 0);
    // The function ModRefSets should include all other locations, however
    EXPECT_EQ(lambdaEntryNodes.size(), pointsToGraph.numMemoryNodes() - 1);
  };

  jlm::llvm::StoreTest1 test;
  // jlm::rvsdg::view(&test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestStore2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::StoreTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto lambdaPtgNode = pointsToGraph.getNodeForLambda(*test.lambda);

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).getModRefNodes();
    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).getModRefNodes();
    EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

    // The memory location representing the function itself should not be included
    EXPECT_EQ(lambdaEntryNodes.count(lambdaPtgNode), 0);
    // The function ModRefSets should include all other locations, however
    EXPECT_EQ(lambdaEntryNodes.size(), pointsToGraph.numMemoryNodes() - 1);
  };

  jlm::llvm::StoreTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestLoad1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::LoadTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto lambdaPtgNode = pointsToGraph.getNodeForLambda(*test.lambda);

    auto & lambdaEntryNodes =
        modRefSummary.GetLambdaEntryModRef(*test.lambda).getModRefNodes();
    auto & lambdaExitNodes =
        modRefSummary.GetLambdaExitModRef(*test.lambda).getModRefNodes();
    EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

    // The memory location representing the function itself should not be included
    EXPECT_EQ(lambdaEntryNodes.count(lambdaPtgNode), 0);
    // The function ModRefSets should include all other locations, however
    EXPECT_EQ(lambdaEntryNodes.size(), pointsToGraph.numMemoryNodes() - 1);
  };

  jlm::llvm::LoadTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestLoad2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::LoadTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto lambdaPtgNode = pointsToGraph.getNodeForLambda(*test.lambda);

    auto & lambdaEntryNodes =
        modRefSummary.GetLambdaEntryModRef(*test.lambda).getModRefNodes();
    auto & lambdaExitNodes =
        modRefSummary.GetLambdaExitModRef(*test.lambda).getModRefNodes();
    EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

    // The memory location representing the function itself should not be included
    EXPECT_EQ(lambdaEntryNodes.count(lambdaPtgNode), 0);
    // The function ModRefSets should include all other locations, however
    EXPECT_EQ(lambdaEntryNodes.size(), pointsToGraph.numMemoryNodes() - 1);
  };

  jlm::llvm::LoadTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestLoadFromUndef)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::LoadFromUndefTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto lambdaPtgNode = pointsToGraph.getNodeForLambda(test.Lambda());

    auto & lambdaEntryNodes =
        modRefSummary.GetLambdaEntryModRef(test.Lambda()).getModRefNodes();
    auto & lambdaExitNodes =
        modRefSummary.GetLambdaExitModRef(test.Lambda()).getModRefNodes();
    EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

    // The memory location representing the function itself should not be included
    EXPECT_EQ(lambdaEntryNodes.count(lambdaPtgNode), 0);
    // The function ModRefSets should include all other locations, however
    EXPECT_EQ(lambdaEntryNodes.size(), pointsToGraph.numMemoryNodes() - 1);
  };

  jlm::llvm::LoadFromUndefTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestCall1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::CallTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto xPtgNode = pointsToGraph.getNodeForAlloca(*test.alloca_x);
    auto yPtgNode = pointsToGraph.getNodeForAlloca(*test.alloca_y);
    auto zPtgNode = pointsToGraph.getNodeForAlloca(*test.alloca_z);
    auto omegaPtgNode = pointsToGraph.getExternalMemoryNode();

    /*
     * Validate function f
     */
    {
      auto & lambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_f).getModRefNodes();
      auto & lambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_f).getModRefNodes();
      EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

      // The f() function should only contain memory affected inside it
      EXPECT_TRUE(lambdaEntryNodes.count(xPtgNode));
      EXPECT_TRUE(lambdaEntryNodes.count(yPtgNode));
      EXPECT_TRUE(lambdaEntryNodes.count(omegaPtgNode));
      EXPECT_EQ(lambdaEntryNodes.size(), 3);
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_g).getModRefNodes();
      auto & lambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_g).getModRefNodes();
      EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

      EXPECT_TRUE(lambdaEntryNodes.count(zPtgNode));
      EXPECT_TRUE(lambdaEntryNodes.count(omegaPtgNode));
      EXPECT_EQ(lambdaEntryNodes.size(), 2);
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_h).getModRefNodes();
      auto & lambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_h).getModRefNodes();
      EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

      EXPECT_TRUE(lambdaEntryNodes.count(xPtgNode));
      EXPECT_TRUE(lambdaEntryNodes.count(yPtgNode));
      EXPECT_TRUE(lambdaEntryNodes.count(zPtgNode));
      EXPECT_TRUE(lambdaEntryNodes.count(omegaPtgNode));
      EXPECT_EQ(lambdaEntryNodes.size(), 4);

      auto & callFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF()).getModRefNodes();
      EXPECT_EQ(callFNodes, lambdaEntryNodes);

      auto & callGNodes = modRefSummary.GetSimpleNodeModRef(test.CallG()).getModRefNodes();
      EXPECT_EQ(callGNodes, lambdaEntryNodes);
    }
  };

  jlm::llvm::CallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestCall2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::CallTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto mallocPtgNode = pointsToGraph.getNodeForAlloca(*test.malloc);
    auto omegaPtgNode = pointsToGraph.getExternalMemoryNode();

    /*
     * Validate function create
     */
    {
      auto lambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_create).getModRefNodes();
      auto lambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_create).getModRefNodes();
      EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

      EXPECT_TRUE(lambdaEntryNodes.count(mallocPtgNode));
      EXPECT_TRUE(lambdaEntryNodes.count(omegaPtgNode));
      EXPECT_EQ(lambdaEntryNodes.size(), 2);
    }

    /*
     * Validate function destroy
     */
    {
      auto lambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_destroy).getModRefNodes();
      auto lambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_destroy).getModRefNodes();
      EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

      EXPECT_TRUE(lambdaEntryNodes.count(mallocPtgNode));
      EXPECT_TRUE(lambdaEntryNodes.count(omegaPtgNode));
      EXPECT_EQ(lambdaEntryNodes.size(), 2);
    }

    /*
     * Validate function test
     */
    {
      auto lambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_test).getModRefNodes();
      auto lambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_test).getModRefNodes();
      EXPECT_EQ(lambdaEntryNodes, lambdaExitNodes);

      // The test() function never loads or stores a pointer to the malloc,
      // so the malloc gets Omega folded
      EXPECT_TRUE(lambdaEntryNodes.count(omegaPtgNode));
      EXPECT_EQ(lambdaEntryNodes.size(), 1);

      auto & callCreate1Nodes =
          modRefSummary.GetSimpleNodeModRef(test.CallCreate1()).getModRefNodes();
      EXPECT_EQ(callCreate1Nodes, lambdaEntryNodes);

      auto & callCreate2Nodes =
          modRefSummary.GetSimpleNodeModRef(test.CallCreate2()).getModRefNodes();
      EXPECT_EQ(callCreate2Nodes, lambdaEntryNodes);

      auto & callDestroy1Nodes =
          modRefSummary.GetSimpleNodeModRef(test.CallDestroy1()).getModRefNodes();
      EXPECT_EQ(callDestroy1Nodes, lambdaEntryNodes);

      auto & callDestroy2Nodes =
          modRefSummary.GetSimpleNodeModRef(test.CallDestroy2()).getModRefNodes();
      EXPECT_EQ(callDestroy2Nodes, lambdaEntryNodes);
    }
  };

  jlm::llvm::CallTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestIndirectCall)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::IndirectCallTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function four
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(test.GetLambdaFour()).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(test.GetLambdaFour()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function three
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree()).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(test.GetLambdaThree()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function indcall
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(test.GetLambdaIndcall()).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(test.GetLambdaIndcall()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());

      auto numCallIndcallNodes =
          modRefSummary.GetSimpleNodeModRef(test.CallIndcall()).getModRefNodes().size();
      EXPECT_EQ(numCallIndcallNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest()).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(test.GetLambdaTest()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());

      auto numCallThreeNodes =
          modRefSummary.GetSimpleNodeModRef(test.CallThree()).getModRefNodes().size();
      EXPECT_EQ(numCallThreeNodes, pointsToGraph.numMemoryNodes());

      auto numCallFourNodes =
          modRefSummary.GetSimpleNodeModRef(test.CallFour()).getModRefNodes().size();
      EXPECT_EQ(numCallFourNodes, pointsToGraph.numMemoryNodes());
    }
  };

  jlm::llvm::IndirectCallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestGamma)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::GammaTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes =
        modRefSummary.GetLambdaEntryModRef(*test.lambda).getModRefNodes().size();
    auto numLambdaExitNodes =
        modRefSummary.GetLambdaExitModRef(*test.lambda).getModRefNodes().size();
    auto numGammaEntryNodes =
        modRefSummary.GetGammaEntryModRef(*test.gamma).getModRefNodes().size();
    auto numGammaExitNodes = modRefSummary.GetGammaExitModRef(*test.gamma).getModRefNodes().size();

    EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numGammaEntryNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numGammaExitNodes, pointsToGraph.numMemoryNodes());
  };

  jlm::llvm::GammaTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestTheta)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::ThetaTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes =
        modRefSummary.GetLambdaEntryModRef(*test.lambda).getModRefNodes().size();
    auto numLambdaExitNodes =
        modRefSummary.GetLambdaExitModRef(*test.lambda).getModRefNodes().size();
    auto numThetaNodes = modRefSummary.GetThetaModRef(*test.theta).getModRefNodes().size();

    EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numThetaNodes, pointsToGraph.numMemoryNodes());
  };

  jlm::llvm::ThetaTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestDelta1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::DeltaTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_g).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_g).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_h).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_h).getModRefNodes().size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallG()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numCallNodes, pointsToGraph.numMemoryNodes());
    }
  };

  jlm::llvm::DeltaTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestDelta2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::DeltaTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f1
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_f1).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_f1).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_f2).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_f2).getModRefNodes().size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numCallNodes, pointsToGraph.numMemoryNodes());
    }
  };

  jlm::llvm::DeltaTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestImports)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::ImportTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f1
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_f1).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_f1).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_f2).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_f2).getModRefNodes().size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numCallNodes, pointsToGraph.numMemoryNodes());
    }
  };

  jlm::llvm::ImportTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*ptg);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestPhi1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::PhiTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function fib
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_fib).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_fib).getModRefNodes().size();
      auto numGammaEntryNodes =
          modRefSummary.GetGammaEntryModRef(*test.gamma).getModRefNodes().size();
      auto numGammaExitNodes =
          modRefSummary.GetGammaExitModRef(*test.gamma).getModRefNodes().size();
      auto numCallFibm1Nodes =
          modRefSummary.GetSimpleNodeModRef(test.CallFibm1()).getModRefNodes().size();
      auto numCallFibm2Nodes =
          modRefSummary.GetSimpleNodeModRef(test.CallFibm2()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numGammaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numGammaExitNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numCallFibm1Nodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numCallFibm2Nodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(*test.lambda_test).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(*test.lambda_test).getModRefNodes().size();
      auto numCallFibNodes =
          modRefSummary.GetSimpleNodeModRef(test.CallFib()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numCallFibNodes, pointsToGraph.numMemoryNodes());
    }
  };

  jlm::llvm::PhiTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestMemcpy)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::MemcpyTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(test.LambdaF()).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(test.LambdaF()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes =
          modRefSummary.GetLambdaEntryModRef(test.LambdaG()).getModRefNodes().size();
      auto numLambdaExitNodes =
          modRefSummary.GetLambdaExitModRef(test.LambdaG()).getModRefNodes().size();
      auto numCallFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF()).getModRefNodes().size();
      auto numMemcpyNodes =
          modRefSummary.GetSimpleNodeModRef(test.Memcpy()).getModRefNodes().size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numCallFNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numMemcpyNodes, 2u);
    }
  };

  jlm::llvm::MemcpyTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

TEST(AgnosticModRefSummarizerTests, TestStatistics)
{
  using namespace jlm::llvm;
  using namespace jlm::llvm::aa;
  using namespace jlm::util;

  // Arrange
  LoadTest1 test;
  auto pointsToGraph = RunAndersen(test.module());

  StatisticsCollectorSettings statisticsCollectorSettings(
      { Statistics::Id::AgnosticModRefSummarizer });
  StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  // Act
  AgnosticModRefSummarizer::Create(test.module(), *pointsToGraph, statisticsCollector);

  // Assert
  EXPECT_EQ(statisticsCollector.NumCollectedStatistics(), 1u);

  auto & statistics = dynamic_cast<const AgnosticModRefSummarizer::Statistics &>(
      *statisticsCollector.CollectedStatistics().begin());

  EXPECT_EQ(statistics.GetSourceFile(), test.module().SourceFileName());
  EXPECT_EQ(statistics.NumPointsToGraphMemoryNodes(), 2u);
  EXPECT_NE(statistics.GetTime(), 0u);
  EXPECT_EQ(
      statistics.GetMeasurementValue<std::string>("MemoryStateDistribution"),
      "LAMBDA[f]-LAMBDA[f]-(0:0)-2-2,LAMBDA[f]-Load[ptr]-(1:0)-2-2,LAMBDA[f]-Load[bit32]-(1:1)-2-"
      "2");
}
