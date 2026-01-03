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
RunAndersen(const jlm::llvm::RvsdgModule & module)
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
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();

    EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
  };

  jlm::llvm::StoreTest1 test;
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

TEST(AgnosticModRefSummarizerTests, TestStore2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::StoreTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();

    EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
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
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();

    EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
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
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();

    EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
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
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.Lambda()).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.Lambda()).Size();

    EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
    EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
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
    /*
     * Validate function f
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());

      auto numCallFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF()).Size();
      EXPECT_EQ(numCallFNodes, pointsToGraph.numMemoryNodes());

      auto numCallGNodes = modRefSummary.GetSimpleNodeModRef(test.CallG()).Size();
      EXPECT_EQ(numCallGNodes, pointsToGraph.numMemoryNodes());
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
    /*
     * Validate function create
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_create).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_create).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function destroy
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_destroy).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_destroy).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());

      auto numCallCreate1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate1()).Size();
      EXPECT_EQ(numCallCreate1Nodes, pointsToGraph.numMemoryNodes());

      auto numCallCreate2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate2()).Size();
      EXPECT_EQ(numCallCreate2Nodes, pointsToGraph.numMemoryNodes());

      auto numCallDestroy1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy1()).Size();
      EXPECT_EQ(numCallDestroy1Nodes, pointsToGraph.numMemoryNodes());

      auto numCallDestroy2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy2()).Size();
      EXPECT_EQ(numCallDestroy2Nodes, pointsToGraph.numMemoryNodes());
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
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaFour()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaFour()).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function three
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree()).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function indcall
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaIndcall()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaIndcall()).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());

      auto numCallIndcallNodes = modRefSummary.GetSimpleNodeModRef(test.CallIndcall()).Size();
      EXPECT_EQ(numCallIndcallNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest()).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());

      auto numCallThreeNodes = modRefSummary.GetSimpleNodeModRef(test.CallThree()).Size();
      EXPECT_EQ(numCallThreeNodes, pointsToGraph.numMemoryNodes());

      auto numCallFourNodes = modRefSummary.GetSimpleNodeModRef(test.CallFour()).Size();
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
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();
    auto numGammaEntryNodes = modRefSummary.GetGammaEntryModRef(*test.gamma).Size();
    auto numGammaExitNodes = modRefSummary.GetGammaExitModRef(*test.gamma).Size();

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
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();
    auto numThetaNodes = modRefSummary.GetThetaModRef(*test.theta).Size();

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
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallG()).Size();

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
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f1).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1()).Size();

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
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f1).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1()).Size();

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
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_fib).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_fib).Size();
      auto numGammaEntryNodes = modRefSummary.GetGammaEntryModRef(*test.gamma).Size();
      auto numGammaExitNodes = modRefSummary.GetGammaExitModRef(*test.gamma).Size();
      auto numCallFibm1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallFibm1()).Size();
      auto numCallFibm2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallFibm2()).Size();

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
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test).Size();
      auto numCallFibNodes = modRefSummary.GetSimpleNodeModRef(test.CallFib()).Size();

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
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaF()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaF()).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaG()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaG()).Size();
      auto numCallFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF()).Size();
      auto numMemcpyNodes = modRefSummary.GetSimpleNodeModRef(test.Memcpy()).Size();

      EXPECT_EQ(numLambdaEntryNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numLambdaExitNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numCallFNodes, pointsToGraph.numMemoryNodes());
      EXPECT_EQ(numMemcpyNodes, 2);
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
  // Arrange
  jlm::llvm::LoadTest1 test;
  auto pointsToGraph = RunAndersen(test.module());

  jlm::util::StatisticsCollectorSettings statisticsCollectorSettings(
      { jlm::util::Statistics::Id::AgnosticModRefSummarizer });
  jlm::util::StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  // Act
  jlm::llvm::aa::AgnosticModRefSummarizer::Create(
      test.module(),
      *pointsToGraph,
      statisticsCollector);

  // Assert
  EXPECT_EQ(statisticsCollector.NumCollectedStatistics(), 1);

  auto & statistics = dynamic_cast<const jlm::llvm::aa::AgnosticModRefSummarizer::Statistics &>(
      *statisticsCollector.CollectedStatistics().begin());

  EXPECT_EQ(statistics.GetSourceFile(), test.module().SourceFileName());
  EXPECT_EQ(statistics.NumPointsToGraphMemoryNodes(), 2);
  EXPECT_NE(statistics.GetTime(), 0);
}
