/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
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
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestStore2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const StoreTest2 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestLoad1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const LoadTest1 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestLoad2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const LoadTest2 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda).Size();

    assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
    assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
  };

  LoadTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());

  /*
   * Act
   */
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestLoadFromUndef()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const LoadFromUndefTest & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.Lambda()).Size();
    auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(test.Lambda()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestCall1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const CallTest1 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_f).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_f).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_g).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_g).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_h).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_h).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallFEntryNodes = provisioning.GetCallEntryNodes(test.CallF()).Size();
      auto numCallFExitNodes = provisioning.GetCallExitNodes(test.CallF()).Size();

      assert(numCallFEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallFExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallGEntryNodes = provisioning.GetCallEntryNodes(test.CallG()).Size();
      auto numCallGExitNodes = provisioning.GetCallExitNodes(test.CallG()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestCall2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const CallTest2 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function create
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_create).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_create).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function destroy
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_destroy).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_destroy).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_test).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_test).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallCreate1EntryNodes = provisioning.GetCallEntryNodes(test.CallCreate1()).Size();
      auto numCallCreate1ExitNodes = provisioning.GetCallExitNodes(test.CallCreate1()).Size();

      assert(numCallCreate1EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallCreate1ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallCreate2EntryNodes = provisioning.GetCallEntryNodes(test.CallCreate2()).Size();
      auto numCallCreate2ExitNodes = provisioning.GetCallExitNodes(test.CallCreate2()).Size();

      assert(numCallCreate2EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallCreate2ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallDestroy1EntryNodes = provisioning.GetCallEntryNodes(test.CallDestroy1()).Size();
      auto numCallDestroy1ExitNodes = provisioning.GetCallExitNodes(test.CallDestroy1()).Size();

      assert(numCallDestroy1EntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallDestroy1ExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallDestroy2EntryNodes = provisioning.GetCallEntryNodes(test.CallDestroy2()).Size();
      auto numCallDestroy2ExitNodes = provisioning.GetCallExitNodes(test.CallDestroy2()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestIndirectCall()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const IndirectCallTest1 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function four
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaFour()).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaFour()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function three
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaThree()).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaThree()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function indcall
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaIndcall()).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaIndcall()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallIndcallEntryNodes = provisioning.GetCallEntryNodes(test.CallIndcall()).Size();
      auto numCallIndcallExitNodes = provisioning.GetCallExitNodes(test.CallIndcall()).Size();

      assert(numCallIndcallEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallIndcallExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function test
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaTest()).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaTest()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallThreeEntryNodes = provisioning.GetCallEntryNodes(test.CallThree()).Size();
      auto numCallThreeExitNodes = provisioning.GetCallExitNodes(test.CallThree()).Size();

      assert(numCallThreeEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numCallThreeExitNodes == pointsToGraph.NumMemoryNodes());

      auto numCallFourEntryNodes = provisioning.GetCallEntryNodes(test.CallFour()).Size();
      auto numCallFourExitNodes = provisioning.GetCallExitNodes(test.CallFour()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestGamma()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const GammaTest & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda).Size();
    auto numGammaEntryNodes = provisioning.GetGammaEntryNodes(*test.gamma).Size();
    auto numGammaExitNodes = provisioning.GetGammaExitNodes(*test.gamma).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestTheta()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const ThetaTest & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda).Size();
    auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda).Size();
    auto numThetaNodes = provisioning.GetThetaEntryExitNodes(*test.theta).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestDelta1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const DeltaTest1 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_g).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_g).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function h
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_h).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_h).Size();
      auto numCallEntryNodes = provisioning.GetCallEntryNodes(test.CallG()).Size();
      auto numCallExitNodes = provisioning.GetCallExitNodes(test.CallG()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestDelta2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const DeltaTest2 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f1
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_f1).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_f1).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_f2).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_f2).Size();
      auto numCallEntryNodes = provisioning.GetCallEntryNodes(test.CallF1()).Size();
      auto numCallExitNodes = provisioning.GetCallExitNodes(test.CallF1()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestImports()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const ImportTest & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f1
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_f1).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_f1).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function f2
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_f2).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_f2).Size();
      auto numCallEntryNodes = provisioning.GetCallEntryNodes(test.CallF1()).Size();
      auto numCallExitNodes = provisioning.GetCallExitNodes(test.CallF1()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestPhi1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const PhiTest1 & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function fib
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_fib).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_fib).Size();
      auto numGammaEntryNodes = provisioning.GetGammaEntryNodes(*test.gamma).Size();
      auto numGammaExitNodes = provisioning.GetGammaExitNodes(*test.gamma).Size();
      auto numCallFibm1EntryNodes = provisioning.GetCallEntryNodes(test.CallFibm1()).Size();
      auto numCallFibm1ExitNodes = provisioning.GetCallExitNodes(test.CallFibm1()).Size();
      auto numCallFibm2EntryNodes = provisioning.GetCallEntryNodes(test.CallFibm2()).Size();
      auto numCallFibm2ExitNodes = provisioning.GetCallExitNodes(test.CallFibm2()).Size();

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
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_test).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_test).Size();
      auto numCallFibEntryNodes = provisioning.GetCallEntryNodes(test.CallFib()).Size();
      auto numCallFibExitNodes = provisioning.GetCallExitNodes(test.CallFib()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestMemcpy()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const MemcpyTest & test,
    const jlm::aa::MemoryNodeProvisioning & provisioning,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function f
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.LambdaF()).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(test.LambdaF()).Size();

      assert(numLambdaEntryNodes == pointsToGraph.NumMemoryNodes());
      assert(numLambdaExitNodes == pointsToGraph.NumMemoryNodes());
    }

    /*
     * Validate function g
     */
    {
      auto numLambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.LambdaG()).Size();
      auto numLambdaExitNodes = provisioning.GetLambdaExitNodes(test.LambdaG()).Size();
      auto numCallFEntryNodes = provisioning.GetCallEntryNodes(test.CallF()).Size();
      auto numCallFExitNodes = provisioning.GetCallExitNodes(test.CallF()).Size();

      auto numMemcpyDestNodes = provisioning.GetOutputNodes(*test.Memcpy().input(0)->origin()).Size();
      auto numMemcpySrcNodes = provisioning.GetOutputNodes(*test.Memcpy().input(1)->origin()).Size();

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
  auto provisioning = jlm::aa::AgnosticMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provisioning, *pointsToGraph);
}

static void
TestStatistics()
{
  /*
   * Arrange
   */
  LoadTest1 test;
  jlm::filepath filePath("/tmp/TestStatistics");
  auto pointsToGraph = RunSteensgaard(test.module());

  jlm::StatisticsCollectorSettings statisticsCollectorSettings(
    filePath,
    {jlm::Statistics::Id::MemoryNodeProvisioning});
  jlm::StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  /*
   * Act
   */
  jlm::aa::AgnosticMemoryNodeProvider::Create(
    test.module(),
    *pointsToGraph,
    statisticsCollector);

  /*
   * Assert
   */
  assert(statisticsCollector.NumCollectedStatistics() == 1);

  auto & statistics = dynamic_cast<const jlm::aa::AgnosticMemoryNodeProvider::Statistics&>(
    *statisticsCollector.CollectedStatistics().begin());

  assert(statistics.NumPointsToGraphMemoryNodes() == 2);
  assert(statistics.GetTime() != 0);
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

  TestStatistics();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAgnosticMemoryNodeProvider", test)