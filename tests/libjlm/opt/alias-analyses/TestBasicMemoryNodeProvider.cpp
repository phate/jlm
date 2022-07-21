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
  StatisticsDescriptor sd;
  return stgd.Analyze(module, sd);
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
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);


  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  LoadTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());

  /*
   * Act
   */
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(test.Lambda()).size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(test.Lambda()).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_g).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_g).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_h).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_h).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallFEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallF()).size();
      auto numCallFExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallF()).size();

      assert(numCallFEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallGEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallG()).size();
      auto numCallGExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallG()).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_create).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_create).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function destroy
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_destroy).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_destroy).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_test).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_test).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallCreate1EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallCreate1()).size();
      auto numCallCreate1ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallCreate1()).size();

      assert(numCallCreate1EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallCreate1ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallCreate2EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallCreate2()).size();
      auto numCallCreate2ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallCreate2()).size();

      assert(numCallCreate2EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallCreate2ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallDestroy1EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallDestroy1()).size();
      auto numCallDestroy1ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallDestroy1()).size();

      assert(numCallDestroy1EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallDestroy1ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallDestroy2EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallDestroy2()).size();
      auto numCallDestroy2ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallDestroy2()).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
}

static void
TestIndirectCall()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const IndirectCallTest & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function four
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_four).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_four).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function three
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_three).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_three).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function indcall
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_indcall).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_indcall).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallIndcallEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallIndcall()).size();
      auto numCallIndcallExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallIndcall()).size();

      assert(numCallIndcallEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallIndcallExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_test).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_test).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallThreeEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallThree()).size();
      auto numCallThreeExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallThree()).size();

      assert(numCallThreeEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallThreeExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallFourEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallFour()).size();
      auto numCallFourExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallFour()).size();

      assert(numCallFourEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFourExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  IndirectCallTest test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).size();
    auto numGammaEntryNodes = basicMemoryNodeProvider.GetGammaEntryNodes(*test.gamma).size();
    auto numGammaExitNodes = basicMemoryNodeProvider.GetGammaExitNodes(*test.gamma).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
    auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda).size();
    auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda).size();
    auto numThetaNodes = basicMemoryNodeProvider.GetThetaEntryExitNodes(*test.theta).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_g).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_g).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_h).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_h).size();
      auto numCallEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallG()).size();
      auto numCallExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallG()).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f1).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f1).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f2).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f2).size();
      auto numCallEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallF1()).size();
      auto numCallExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallF1()).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f1).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f1).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_f2).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_f2).size();
      auto numCallEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallF1()).size();
      auto numCallExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallF1()).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
}

static void
TestPhi()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const PhiTest & test,
    const jlm::aa::BasicMemoryNodeProvider & basicMemoryNodeProvider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function fib
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_fib).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_fib).size();
      auto numGammaEntryNodes = basicMemoryNodeProvider.GetGammaEntryNodes(*test.gamma).size();
      auto numGammaExitNodes = basicMemoryNodeProvider.GetGammaExitNodes(*test.gamma).size();
      auto numCallFibm1EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallFibm1()).size();
      auto numCallFibm1ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallFibm1()).size();
      auto numCallFibm2EntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallFibm2()).size();
      auto numCallFibm2ExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallFibm2()).size();

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
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(*test.lambda_test).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(*test.lambda_test).size();
      auto numCallFibEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallFib()).size();
      auto numCallFibExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallFib()).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFibExitNodes == pointsToGraph.NumMemoryNodes());
    }
  };

  PhiTest test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(test.LambdaF()).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(test.LambdaF()).size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = basicMemoryNodeProvider.GetLambdaEntryNodes(test.LambdaG()).size();
      auto numLambdaExitNodes = basicMemoryNodeProvider.GetLambdaExitNodes(test.LambdaG()).size();
      auto numCallFEntryNodes = basicMemoryNodeProvider.GetCallEntryNodes(test.CallF()).size();
      auto numCallFExitNodes = basicMemoryNodeProvider.GetCallExitNodes(test.CallF()).size();

      auto numMemcpyDestNodes = basicMemoryNodeProvider.GetOutputNodes(*test.Memcpy().input(0)->origin()).size();
      auto numMemcpySrcNodes = basicMemoryNodeProvider.GetOutputNodes(*test.Memcpy().input(1)->origin()).size();

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
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(*pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, basicMemoryNodeProvider, *pointsToGraph);
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

  TestPhi();

  TestMemcpy();

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/TestBasicMemoryNodeProvider", test)