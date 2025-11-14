/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

#include <iostream>

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunAndersen(const jlm::llvm::RvsdgModule & module)
{
  using namespace jlm::llvm;

  aa::Andersen andersen;
  jlm::util::StatisticsCollector statisticsCollector;
  return andersen.Analyze(module, statisticsCollector);
}

static void
TestStore1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::StoreTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  jlm::tests::StoreTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestStore2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::StoreTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  jlm::tests::StoreTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestLoad1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::LoadTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  jlm::tests::LoadTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestLoad2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::LoadTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  jlm::tests::LoadTest2 test;
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

static void
TestLoadFromUndef()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::LoadFromUndefTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.Lambda()).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.Lambda()).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  jlm::tests::LoadFromUndefTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

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

static void
TestCall1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::CallTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF()).Size();
      assert(numCallFNodes == pointsToGraph.NumMemoryNodes());

      auto numCallGNodes = modRefSummary.GetSimpleNodeModRef(test.CallG()).Size();
      assert(numCallGNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  jlm::tests::CallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestCall2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::CallTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function create
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_create).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_create).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function destroy
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_destroy).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_destroy).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallCreate1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate1()).Size();
      assert(numCallCreate1Nodes == pointsToGraph.NumMemoryNodes());

      auto numCallCreate2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate2()).Size();
      assert(numCallCreate2Nodes == pointsToGraph.NumMemoryNodes());

      auto numCallDestroy1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy1()).Size();
      assert(numCallDestroy1Nodes == pointsToGraph.NumMemoryNodes());

      auto numCallDestroy2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy2()).Size();
      assert(numCallDestroy2Nodes == pointsToGraph.NumMemoryNodes());
    }
  };

  jlm::tests::CallTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestIndirectCall()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::IndirectCallTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function four
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaFour()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaFour()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function three
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function indcall
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaIndcall()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaIndcall()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallIndcallNodes = modRefSummary.GetSimpleNodeModRef(test.CallIndcall()).Size();
      assert(numCallIndcallNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallThreeNodes = modRefSummary.GetSimpleNodeModRef(test.CallThree()).Size();
      assert(numCallThreeNodes == pointsToGraph.NumMemoryNodes());

      auto numCallFourNodes = modRefSummary.GetSimpleNodeModRef(test.CallFour()).Size();
      assert(numCallFourNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  jlm::tests::IndirectCallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestGamma()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::GammaTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();
    auto numGammaEntryNodes = modRefSummary.GetGammaEntryModRef(*test.gamma).Size();
    auto numGammaExitNodes = modRefSummary.GetGammaExitModRef(*test.gamma).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    assert(numGammaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numGammaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  jlm::tests::GammaTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestTheta()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::ThetaTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda).Size();
    auto numThetaNodes = modRefSummary.GetThetaModRef(*test.theta).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    assert(numThetaNodes == pointsToGraph.NumMemoryNodes());
  };

  jlm::tests::ThetaTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestDelta1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::DeltaTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallG()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  jlm::tests::DeltaTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestDelta2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::DeltaTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f1
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f1).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  jlm::tests::DeltaTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestImports()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::ImportTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f1
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f1).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  jlm::tests::ImportTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*ptg);

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

static void
TestPhi1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::PhiTest1 & test,
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

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numGammaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numGammaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibm1Nodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibm2Nodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test).Size();
      auto numCallFibNodes = modRefSummary.GetSimpleNodeModRef(test.CallFib()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  jlm::tests::PhiTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestMemcpy()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::MemcpyTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaF()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaF()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaG()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaG()).Size();
      auto numCallFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF()).Size();
      auto numMemcpyNodes = modRefSummary.GetSimpleNodeModRef(test.Memcpy()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFNodes == pointsToGraph.NumMemoryNodes());
      assert(numMemcpyNodes == 2);
    }
  };

  jlm::tests::MemcpyTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

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

static void
TestStatistics()
{
  // Arrange
  jlm::tests::LoadTest1 test;
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
  assert(statisticsCollector.NumCollectedStatistics() == 1);

  auto & statistics = dynamic_cast<const jlm::llvm::aa::AgnosticModRefSummarizer::Statistics &>(
      *statisticsCollector.CollectedStatistics().begin());

  assert(statistics.GetSourceFile() == test.module().SourceFileName());
  assert(statistics.NumPointsToGraphMemoryNodes() == 2);
  assert(statistics.GetTime() != 0);
}

static void
test()
{
  TestStore1();
  TestStore2();

  TestLoad1();
  TestLoad2();
  TestLoadFromUndef();

  TestCall1();
  TestCall2();
  TestIndirectCall();

  TestGamma();
  TestTheta();

  TestDelta1();
  TestDelta2();

  TestImports();

  TestPhi1();

  TestMemcpy();

  TestStatistics();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizerTests", test)
