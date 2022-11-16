/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "AliasAnalysesTests.hpp"

#include <test-registry.hpp>

#include <jive/view.hpp>

#include <jlm/opt/alias-analyses/BasicMemoryNodeProvider.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/util/Statistics.hpp>

#include <iostream>

static std::unique_ptr<jlm::aa::PointsToGraph>
RunSteensgaard(const jlm::RvsdgModule & module)
{
  using namespace jlm;

  aa::Steensgaard stgd;
  StatisticsCollector statisticsCollector;
  return stgd.Analyze(module, statisticsCollector);
}

static void
TestStore1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const StoreTest1 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  StoreTest1 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestStore2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const StoreTest2 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  StoreTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestLoad1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const LoadTest1 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  LoadTest1 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph= RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestLoad2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const LoadTest2 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  LoadTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestLoadFromUndef()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const LoadFromUndefTest & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(test.Lambda()).Size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(test.Lambda()).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  LoadFromUndefTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestCall1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const CallTest1 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_g).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_g).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_h).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_h).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallFEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallF()).Size();
      auto numCallFExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallF()).Size();

      assert(numCallFEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallGEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallG()).Size();
      auto numCallGExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallG()).Size();

      assert(numCallGEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallGExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  CallTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestCall2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const CallTest2 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function create
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_create).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_create).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function destroy
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_destroy).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_destroy).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_test).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_test).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallCreate1EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallCreate1()).Size();
      auto numCallCreate1ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallCreate1()).Size();

      assert(numCallCreate1EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallCreate1ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallCreate2EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallCreate2()).Size();
      auto numCallCreate2ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallCreate2()).Size();

      assert(numCallCreate2EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallCreate2ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallDestroy1EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallDestroy1()).Size();
      auto numCallDestroy1ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallDestroy1()).Size();

      assert(numCallDestroy1EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallDestroy1ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallDestroy2EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallDestroy2()).Size();
      auto numCallDestroy2ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallDestroy2()).Size();

      assert(numCallDestroy2EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallDestroy2ExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  CallTest2 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestIndirectCall()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const IndirectCallTest1 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function four
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_four).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_four).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function three
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_three).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_three).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function indcall
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_indcall).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_indcall).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallIndcallEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallIndcall()).Size();
      auto numCallIndcallExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallIndcall()).Size();

      assert(numCallIndcallEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallIndcallExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_test).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_test).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallThreeEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallThree()).Size();
      auto numCallThreeExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallThree()).Size();

      assert(numCallThreeEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallThreeExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallFourEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallFour()).Size();
      auto numCallFourExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallFour()).Size();

      assert(numCallFourEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFourExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  IndirectCallTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestGamma()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const GammaTest & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).Size();
    auto numGammaEntryNodes = basicMemoryNodeProvider.GetGammaEntryNodes(*test.gamma).Size();
    auto numGammaExitNodes = basicMemoryNodeProvider.GetGammaExitNodes(*test.gamma).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    assert(numGammaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numGammaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  GammaTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestTheta()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const ThetaTest & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).Size();
    auto numThetaNodes = basicMemoryNodeProvider.GetThetaEntryExitNodes(*test.theta).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    assert(numThetaNodes == pointsToGraph.NumMemoryNodes());
  };

  ThetaTest test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestDelta1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const DeltaTest1 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_g).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_g).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_h).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_h).Size();
      auto numCallEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallG()).Size();
      auto numCallExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallG()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  DeltaTest1 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestDelta2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const DeltaTest2 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f1
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f1).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f1).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f2).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f2).Size();
      auto numCallEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallF1()).Size();
      auto numCallExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallF1()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  DeltaTest2 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestImports()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const ImportTest & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f1
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f1).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f1).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f2).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f2).Size();
      auto numCallEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallF1()).Size();
      auto numCallExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallF1()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  ImportTest test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*ptg);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestPhi1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const PhiTest1 & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function fib
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_fib).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_fib).Size();
      auto numGammaEntryNodes = basicMemoryNodeProvider.GetGammaEntryNodes(*test.gamma).Size();
      auto numGammaExitNodes = basicMemoryNodeProvider.GetGammaExitNodes(*test.gamma).Size();
      auto numCallFibm1EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallFibm1()).Size();
      auto numCallFibm1ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallFibm1()).Size();
      auto numCallFibm2EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallFibm2()).Size();
      auto numCallFibm2ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallFibm2()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numGammaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numGammaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibm1EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibm1ExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibm2EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibm2ExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_test).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_test).Size();
      auto numCallFibEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallFib()).Size();
      auto numCallFibExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallFib()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  PhiTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestMemcpy()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const MemcpyTest & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(test.LambdaF()).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(test.LambdaF()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(test.LambdaG()).Size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(test.LambdaG()).Size();
      auto numCallFEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallF()).Size();
      auto numCallFExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallF()).Size();

      auto numMemcpyDestNodes = basicMemoryNodeProvider.GetOutputNodes(*test.Memcpy().input(0)->origin()).Size();
      auto numMemcpySrcNodes = basicMemoryNodeProvider.GetOutputNodes(*test.Memcpy().input(1)->origin()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numMemcpyDestNodes == 1);
      assert(numMemcpySrcNodes == 1);
    }
  };

  MemcpyTest test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::BasicMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}


static int
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

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/TestBasicMemoryNodeProvider", test)