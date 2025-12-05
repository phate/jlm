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

    assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
  };

  jlm::tests::StoreTest1 test;
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

    assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
  };

  jlm::tests::StoreTest2 test;
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

    assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
  };

  jlm::tests::LoadTest1 test;
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

    assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
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

    assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
  };

  jlm::tests::LoadFromUndefTest test;
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

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());

      auto numCallFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF()).Size();
      assert(numCallFNodes == pointsToGraph.numMemoryNodes());

      auto numCallGNodes = modRefSummary.GetSimpleNodeModRef(test.CallG()).Size();
      assert(numCallGNodes == pointsToGraph.numMemoryNodes());
    }
  };

  jlm::tests::CallTest1 test;
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

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function destroy
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_destroy).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_destroy).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());

      auto numCallCreate1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate1()).Size();
      assert(numCallCreate1Nodes == pointsToGraph.numMemoryNodes());

      auto numCallCreate2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate2()).Size();
      assert(numCallCreate2Nodes == pointsToGraph.numMemoryNodes());

      auto numCallDestroy1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy1()).Size();
      assert(numCallDestroy1Nodes == pointsToGraph.numMemoryNodes());

      auto numCallDestroy2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy2()).Size();
      assert(numCallDestroy2Nodes == pointsToGraph.numMemoryNodes());
    }
  };

  jlm::tests::CallTest2 test;
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

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function three
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function indcall
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaIndcall()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaIndcall()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());

      auto numCallIndcallNodes = modRefSummary.GetSimpleNodeModRef(test.CallIndcall()).Size();
      assert(numCallIndcallNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());

      auto numCallThreeNodes = modRefSummary.GetSimpleNodeModRef(test.CallThree()).Size();
      assert(numCallThreeNodes == pointsToGraph.numMemoryNodes());

      auto numCallFourNodes = modRefSummary.GetSimpleNodeModRef(test.CallFour()).Size();
      assert(numCallFourNodes == pointsToGraph.numMemoryNodes());
    }
  };

  jlm::tests::IndirectCallTest1 test;
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

    assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    assert(numGammaEntryNodes == pointsToGraph.numMemoryNodes());
    assert(numGammaExitNodes == pointsToGraph.numMemoryNodes());
  };

  jlm::tests::GammaTest test;
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

    assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    assert(numThetaNodes == pointsToGraph.numMemoryNodes());
  };

  jlm::tests::ThetaTest test;
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

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallG()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
      assert(numCallNodes == pointsToGraph.numMemoryNodes());
    }
  };

  jlm::tests::DeltaTest1 test;
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

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
      assert(numCallNodes == pointsToGraph.numMemoryNodes());
    }
  };

  jlm::tests::DeltaTest2 test;
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

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2).Size();
      auto numCallNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
      assert(numCallNodes == pointsToGraph.numMemoryNodes());
    }
  };

  jlm::tests::ImportTest test;
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

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
      assert(numGammaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numGammaExitNodes == pointsToGraph.numMemoryNodes());
      assert(numCallFibm1Nodes == pointsToGraph.numMemoryNodes());
      assert(numCallFibm2Nodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test).Size();
      auto numCallFibNodes = modRefSummary.GetSimpleNodeModRef(test.CallFib()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
      assert(numCallFibNodes == pointsToGraph.numMemoryNodes());
    }
  };

  jlm::tests::PhiTest1 test;
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

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaG()).Size();
      auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaG()).Size();
      auto numCallFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF()).Size();
      auto numMemcpyNodes = modRefSummary.GetSimpleNodeModRef(test.Memcpy()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.numMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.numMemoryNodes());
      assert(numCallFNodes == pointsToGraph.numMemoryNodes());
      assert(numMemcpyNodes == 2);
    }
  };

  jlm::tests::MemcpyTest test;
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
