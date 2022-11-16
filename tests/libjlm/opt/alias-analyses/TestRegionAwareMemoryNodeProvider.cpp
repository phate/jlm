/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "AliasAnalysesTests.hpp"

#include <test-registry.hpp>

#include <jlm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/util/Statistics.hpp>

#include <jive/view.hpp>

static std::unique_ptr<jlm::aa::PointsToGraph>
RunSteensgaard(jlm::RvsdgModule & rvsdgModule)
{
  using namespace jlm;

  aa::Steensgaard steensgaard;
  StatisticsCollector statisticsCollector;
  return steensgaard.Analyze(rvsdgModule, statisticsCollector);
}

static void
AssertMemoryNodes(
  const jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> & receivedMemoryNodes,
  const jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> & expectedMemoryNodes)
{
  assert(receivedMemoryNodes == expectedMemoryNodes);
}

static void
TestStore1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const StoreTest1 & test,
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & allocaAMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_a);
    auto & allocaBMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_b);
    auto & allocaCMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_c);
    auto & allocaDMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_d);

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes(
      {
        &allocaAMemoryNode,
        &allocaBMemoryNode,
        &allocaCMemoryNode,
        &allocaDMemoryNode,
      });

    auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  StoreTest1 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & allocaAMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_a);
    auto & allocaBMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_b);
    auto & allocaPMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_p);
    auto & allocaXMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_x);
    auto & allocaYMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_y);

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes(
      {
        &allocaAMemoryNode,
        &allocaBMemoryNode,
        &allocaPMemoryNode,
        &allocaXMemoryNode,
        &allocaYMemoryNode
      });

    auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  StoreTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, {&lambdaMemoryNode, &externalMemoryNode});

    auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, {&lambdaMemoryNode, &externalMemoryNode});
  };

  LoadTest1 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & allocaAMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_a);
    auto & allocaBMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_b);
    auto & allocaPMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_p);
    auto & allocaXMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_x);
    auto & allocaYMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_y);

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes(
      {
        &allocaAMemoryNode,
        &allocaBMemoryNode,
        &allocaPMemoryNode,
        &allocaXMemoryNode,
        &allocaYMemoryNode
      });

    auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  LoadTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto numLambdaEntryNodes = provider.GetLambdaEntryNodes(test.Lambda()).Size();
    auto numLambdaExitNodes = provider.GetLambdaExitNodes(test.Lambda()).Size();

    assert(numLambdaEntryNodes == 0);
    assert(numLambdaExitNodes == 0);
  };

  LoadFromUndefTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & allocaXMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_x);
    auto & allocaYMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_y);
    auto & allocaZMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_z);

    /*
     * Validate function f
     */
    {
      auto & lambdaFEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_f);
      AssertMemoryNodes(lambdaFEntryNodes, {&allocaXMemoryNode, &allocaYMemoryNode});

      auto & lambdaFExitNodes = provider.GetLambdaExitNodes(*test.lambda_f);
      AssertMemoryNodes(lambdaFExitNodes, {&allocaXMemoryNode, &allocaYMemoryNode});
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaGEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_g);
      AssertMemoryNodes(lambdaGEntryNodes, {&allocaZMemoryNode});

      auto & lambdaGExitNodes = provider.GetLambdaExitNodes(*test.lambda_g);
      AssertMemoryNodes(lambdaGExitNodes, {&allocaZMemoryNode});
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaHEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_h);
      AssertMemoryNodes(lambdaHEntryNodes, {&allocaXMemoryNode, &allocaYMemoryNode, &allocaZMemoryNode});

      auto & callFEntryNodes = provider.GetCallEntryNodes(test.CallF());
      AssertMemoryNodes(callFEntryNodes, {&allocaXMemoryNode, &allocaYMemoryNode});

      auto & callFExitNodes = provider.GetCallExitNodes(test.CallF());
      AssertMemoryNodes(callFExitNodes, {&allocaXMemoryNode, &allocaYMemoryNode});

      auto & callGEntryNodes = provider.GetCallEntryNodes(test.CallG());
      AssertMemoryNodes(callGEntryNodes, {&allocaZMemoryNode});

      auto & callGExitNodes = provider.GetCallExitNodes(test.CallG());
      AssertMemoryNodes(callGExitNodes, {&allocaZMemoryNode});

      auto & lambdaHExitNodes = provider.GetLambdaExitNodes(*test.lambda_h);
      AssertMemoryNodes(lambdaHExitNodes, {&allocaXMemoryNode, &allocaYMemoryNode, &allocaZMemoryNode});
    }
  };

  CallTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & mallocMemoryNode = pointsToGraph.GetMallocNode(*test.malloc);

    /*
     * Validate function create
     */
    {
      auto & lambdaCreateEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_create);
      AssertMemoryNodes(lambdaCreateEntryNodes, {&mallocMemoryNode});

      auto & lambdaCreateExitNodes = provider.GetLambdaExitNodes(*test.lambda_create);
      AssertMemoryNodes(lambdaCreateExitNodes, {&mallocMemoryNode});
    }

    /*
     * Validate function destroy
     */
    {
      auto & lambdaDestroyEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_destroy);
      AssertMemoryNodes(lambdaDestroyEntryNodes, {&mallocMemoryNode});

      auto & lambdaDestroyExitNodes = provider.GetLambdaExitNodes(*test.lambda_destroy);
      AssertMemoryNodes(lambdaDestroyExitNodes, {&mallocMemoryNode});
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaTestEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaTestEntryNodes, {&mallocMemoryNode});

      auto & callCreate1EntryNodes = provider.GetCallEntryNodes(test.CallCreate1());
      AssertMemoryNodes(callCreate1EntryNodes, {&mallocMemoryNode});

      auto & callCreate1ExitNodes = provider.GetCallExitNodes(test.CallCreate1());
      AssertMemoryNodes(callCreate1ExitNodes, {&mallocMemoryNode});

      auto & callCreate2EntryNodes = provider.GetCallEntryNodes(test.CallCreate2());
      AssertMemoryNodes(callCreate2EntryNodes, {&mallocMemoryNode});

      auto & callCreate2ExitNodes = provider.GetCallExitNodes(test.CallCreate2());
      AssertMemoryNodes(callCreate2ExitNodes, {&mallocMemoryNode});

      auto & callDestroy1EntryNodes = provider.GetCallEntryNodes(test.CallDestroy1());
      AssertMemoryNodes(callDestroy1EntryNodes, {&mallocMemoryNode});

      auto & callDestroy1ExitNodes = provider.GetCallExitNodes(test.CallDestroy1());
      AssertMemoryNodes(callDestroy1ExitNodes, {&mallocMemoryNode});

      auto & callDestroy2EntryNodes = provider.GetCallEntryNodes(test.CallDestroy2());
      AssertMemoryNodes(callDestroy2EntryNodes, {&mallocMemoryNode});

      auto & callDestroy2ExitNodes = provider.GetCallExitNodes(test.CallDestroy2());
      AssertMemoryNodes(callDestroy2ExitNodes, {&mallocMemoryNode});

      auto & lambdaTestExitNodes = provider.GetLambdaExitNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaTestExitNodes, {&mallocMemoryNode});
    }
  };

  CallTest2 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    /*
     * Validate function four
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_four);
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_four);
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function three
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_three);
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_three);
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function indcall
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_indcall);
      AssertMemoryNodes(lambdaEntryNodes, {&externalMemoryNode});

      auto & callEntryNodes = provider.GetCallEntryNodes(test.CallIndcall());
      AssertMemoryNodes(callEntryNodes, {&externalMemoryNode});

      auto & callExitNodes = provider.GetCallExitNodes(test.CallIndcall());
      AssertMemoryNodes(callExitNodes, {&externalMemoryNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_indcall);
      AssertMemoryNodes(lambdaExitNodes, {&externalMemoryNode});
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaEntryNodes, {&externalMemoryNode});

      auto & callFourEntryNodes = provider.GetCallEntryNodes(test.CallFour());
      AssertMemoryNodes(callFourEntryNodes, {&externalMemoryNode});

      auto & callFourExitNodes = provider.GetCallExitNodes(test.CallFour());
      AssertMemoryNodes(callFourExitNodes, {&externalMemoryNode});

      auto & callThreeEntryNodes = provider.GetCallEntryNodes(test.CallThree());
      AssertMemoryNodes(callThreeEntryNodes, {&externalMemoryNode});

      auto & callThreeExitNodes = provider.GetCallExitNodes(test.CallThree());
      AssertMemoryNodes(callThreeExitNodes, {&externalMemoryNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaExitNodes, {&externalMemoryNode});
    }
  };

  IndirectCallTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestIndirectCall2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const IndirectCallTest2 & test,
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & deltaG1MemoryNode = pointsToGraph.GetDeltaNode(test.GetDeltaG1());
    auto & deltaG2MemoryNode = pointsToGraph.GetDeltaNode(test.GetDeltaG2());

    auto & allocaPxMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPx());
    auto & allocaPyMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPy());
    auto & allocaPzMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPz());

    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes(
      {
        &deltaG1MemoryNode,
        &deltaG2MemoryNode,
        &allocaPxMemoryNode,
        &allocaPyMemoryNode,
        &allocaPzMemoryNode,
        &externalMemoryNode
      });

    /*
     * Validate function four()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaFour());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaFour());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function three()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaThree());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaThree());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaI());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = provider.GetCallEntryNodes(test.GetIndirectCall());
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = provider.GetCallExitNodes(test.GetIndirectCall());
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaI());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function x()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaX());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaX());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function y()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaY());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaY());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callXEntryNodes = provider.GetCallEntryNodes(test.GetTestCallX());
      AssertMemoryNodes(callXEntryNodes, expectedMemoryNodes);

      auto & callXExitNodes = provider.GetCallExitNodes(test.GetTestCallX());
      AssertMemoryNodes(callXExitNodes, expectedMemoryNodes);

      auto & callYEntryNodes = provider.GetCallEntryNodes(test.GetCallY());
      AssertMemoryNodes(callYEntryNodes, expectedMemoryNodes);

      auto & callYExitNodes = provider.GetCallExitNodes(test.GetCallY());
      AssertMemoryNodes(callYExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function test2()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaTest2());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callXEntryNodes = provider.GetCallEntryNodes(test.GetTest2CallX());
      AssertMemoryNodes(callXEntryNodes, expectedMemoryNodes);

      auto & callXExitNodes = provider.GetCallExitNodes(test.GetTest2CallX());
      AssertMemoryNodes(callXExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaTest2());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }
  };

  IndirectCallTest2 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, {&lambdaMemoryNode, &externalMemoryNode});

    auto gammaEntryNodes = provider.GetGammaEntryNodes(*test.gamma);
    AssertMemoryNodes(gammaEntryNodes, {});

    auto gammaExitNodes = provider.GetGammaExitNodes(*test.gamma);
    AssertMemoryNodes(gammaExitNodes, {});

    auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, {&lambdaMemoryNode, &externalMemoryNode});
  };

  GammaTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, {&lambdaMemoryNode, &externalMemoryNode});

    auto & thetaEntryExitNodes = provider.GetThetaEntryExitNodes(*test.theta);
    AssertMemoryNodes(thetaEntryExitNodes, {&lambdaMemoryNode, &externalMemoryNode});

    auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, {&lambdaMemoryNode, &externalMemoryNode});
  };

  ThetaTest test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & deltaFNode = pointsToGraph.GetDeltaNode(*test.delta_f);

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_g);
      AssertMemoryNodes(lambdaEntryNodes, {&deltaFNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_g);
      AssertMemoryNodes(lambdaExitNodes, {&deltaFNode});
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_h);
      AssertMemoryNodes(lambdaEntryNodes, {&deltaFNode});

      auto & callEntryNodes = provider.GetCallEntryNodes(test.CallG());
      AssertMemoryNodes(callEntryNodes, {&deltaFNode});

      auto & callExitNodes = provider.GetCallExitNodes(test.CallG());
      AssertMemoryNodes(callExitNodes, {&deltaFNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_h);
      AssertMemoryNodes(lambdaExitNodes, {&deltaFNode});
    }
  };

  DeltaTest1 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & deltaD1Node = pointsToGraph.GetDeltaNode(*test.delta_d1);
    auto & deltaD2Node = pointsToGraph.GetDeltaNode(*test.delta_d2);

    /*
     * Validate function f1
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_f1);
      AssertMemoryNodes(lambdaEntryNodes, {&deltaD1Node});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_f1);
      AssertMemoryNodes(lambdaExitNodes, {&deltaD1Node});
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_f2);
      AssertMemoryNodes(lambdaEntryNodes, {&deltaD1Node, &deltaD2Node});

      auto & callEntryNodes = provider.GetCallEntryNodes(test.CallF1());
      AssertMemoryNodes(callEntryNodes, {&deltaD1Node});

      auto & callExitNodes = provider.GetCallExitNodes(test.CallF1());
      AssertMemoryNodes(callExitNodes, {&deltaD1Node});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_f2);
      AssertMemoryNodes(lambdaExitNodes, {&deltaD1Node, &deltaD2Node});
    }
  };

  DeltaTest2 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & importD1Node = pointsToGraph.GetImportNode(*test.import_d1);
    auto & importD2Node = pointsToGraph.GetImportNode(*test.import_d2);

    /*
     * Validate function f1
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_f1);
      AssertMemoryNodes(lambdaEntryNodes, {&importD1Node});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_f1);
      AssertMemoryNodes(lambdaExitNodes, {&importD1Node});
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_f2);
      AssertMemoryNodes(lambdaEntryNodes, {&importD1Node, &importD2Node});

      auto & callEntryNodes = provider.GetCallEntryNodes(test.CallF1());
      AssertMemoryNodes(callEntryNodes, {&importD1Node});

      auto & callExitNodes = provider.GetCallExitNodes(test.CallF1());
      AssertMemoryNodes(callExitNodes, {&importD1Node});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_f2);
      AssertMemoryNodes(lambdaExitNodes, {&importD1Node, &importD2Node});
    }
  };

  ImportTest test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & resultAllocaNode = pointsToGraph.GetAllocaNode(*test.alloca);

    /*
     * Validate function fib
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_fib);
      AssertMemoryNodes(lambdaEntryNodes, {&resultAllocaNode});

      auto & callFibM1EntryNodes = provider.GetCallEntryNodes(test.CallFibm1());
      AssertMemoryNodes(callFibM1EntryNodes, {&resultAllocaNode});

      auto & callFibM1ExitNodes = provider.GetCallExitNodes(test.CallFibm1());
      AssertMemoryNodes(callFibM1ExitNodes, {&resultAllocaNode});

      auto & callFibM2EntryNodes = provider.GetCallEntryNodes(test.CallFibm2());
      AssertMemoryNodes(callFibM2EntryNodes, {&resultAllocaNode});

      auto & callFibM2ExitNodes = provider.GetCallExitNodes(test.CallFibm2());
      AssertMemoryNodes(callFibM2ExitNodes, {&resultAllocaNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_fib);
      AssertMemoryNodes(lambdaExitNodes, {&resultAllocaNode});
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaEntryNodes, {&resultAllocaNode});

      auto & callEntryNodes = provider.GetCallEntryNodes(test.CallFib());
      AssertMemoryNodes(callEntryNodes, {&resultAllocaNode});

      auto & callExitNodes = provider.GetCallExitNodes(test.CallFib());
      AssertMemoryNodes(callExitNodes, {&resultAllocaNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaExitNodes, {&resultAllocaNode});
    }
  };

  PhiTest1 test;
	// jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestPhi2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const PhiTest2 & test,
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & pTestAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPTestAlloca());
    auto & paAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPaAlloca());
    auto & pbAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPbAlloca());
    auto & pcAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPcAlloca());
    auto & pdAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPdAlloca());

    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes(
      {
        &pTestAllocaMemoryNode,
        &paAllocaMemoryNode,
        &pbAllocaMemoryNode,
        &pcAllocaMemoryNode,
        &pdAllocaMemoryNode,
        &externalMemoryNode
      });

    /*
     * Validate function eight()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaEight());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaEight());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaI());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = provider.GetCallEntryNodes(test.GetIndirectCall());
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = provider.GetCallExitNodes(test.GetIndirectCall());
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaI());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function a()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaA());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callBEntryNodes = provider.GetCallEntryNodes(test.GetCallB());
      AssertMemoryNodes(callBEntryNodes, expectedMemoryNodes);

      auto & callBExitNodes = provider.GetCallExitNodes(test.GetCallB());
      AssertMemoryNodes(callBExitNodes, expectedMemoryNodes);

      auto & callDEntryNodes = provider.GetCallEntryNodes(test.GetCallD());
      AssertMemoryNodes(callDEntryNodes, expectedMemoryNodes);

      auto & callDExitNodes = provider.GetCallExitNodes(test.GetCallD());
      AssertMemoryNodes(callDExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaA());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function b()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaB());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callIEntryNodes = provider.GetCallEntryNodes(test.GetCallI());
      AssertMemoryNodes(callIEntryNodes, expectedMemoryNodes);

      auto & callIExitNodes = provider.GetCallExitNodes(test.GetCallI());
      AssertMemoryNodes(callIExitNodes, expectedMemoryNodes);

      auto & callCEntryNodes = provider.GetCallEntryNodes(test.GetCallC());
      AssertMemoryNodes(callCEntryNodes, expectedMemoryNodes);

      auto & callCExitNodes = provider.GetCallExitNodes(test.GetCallC());
      AssertMemoryNodes(callCExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaB());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function c()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaC());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = provider.GetCallEntryNodes(test.GetCallAFromC());
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = provider.GetCallExitNodes(test.GetCallAFromC());
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaC());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function d()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaD());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = provider.GetCallEntryNodes(test.GetCallAFromD());
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = provider.GetCallExitNodes(test.GetCallAFromD());
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaD());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = provider.GetCallEntryNodes(test.GetCallAFromTest());
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = provider.GetCallExitNodes(test.GetCallAFromTest());
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }
  };

  PhiTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

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
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & localArrayMemoryNode = pointsToGraph.GetDeltaNode(test.LocalArray());
    auto & globalArrayMemoryNode = pointsToGraph.GetDeltaNode(test.GlobalArray());

    /*
     * Validate function f
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.LambdaF());
      AssertMemoryNodes(lambdaEntryNodes, {&globalArrayMemoryNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.LambdaF());
      AssertMemoryNodes(lambdaExitNodes, {&globalArrayMemoryNode});
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(test.LambdaG());
      AssertMemoryNodes(lambdaEntryNodes, {&localArrayMemoryNode, &globalArrayMemoryNode});

      auto & callEntryNodes = provider.GetCallEntryNodes(test.CallF());
      AssertMemoryNodes(callEntryNodes, {&globalArrayMemoryNode});

      auto & callExitNodes = provider.GetCallExitNodes(test.CallF());
      AssertMemoryNodes(callExitNodes, {&globalArrayMemoryNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(test.LambdaG());
      AssertMemoryNodes(lambdaExitNodes, {&localArrayMemoryNode, &globalArrayMemoryNode});
    }
  };

  MemcpyTest test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestEscapedMemory1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const EscapedMemoryTest1 & test,
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto & deltaAMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaA);
    auto & deltaBMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaB);
    auto & deltaXMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaX);
    auto & deltaYMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaY);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes({
      &lambdaMemoryNode,
      &deltaAMemoryNode,
      &deltaBMemoryNode,
      &deltaXMemoryNode,
      &deltaYMemoryNode,
      &externalMemoryNode});

    auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.LambdaTest);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.LambdaTest);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  EscapedMemoryTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestEscapedMemory2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const EscapedMemoryTest2 & test,
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & returnAddressMallocMemoryNode = pointsToGraph.GetMallocNode(*test.ReturnAddressMalloc);
    auto & callExternalFunction1MallocMemoryNode = pointsToGraph.GetMallocNode(*test.CallExternalFunction1Malloc);

    auto & returnAddressLambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.ReturnAddressFunction);
    auto & callExternalFunction1LambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.CallExternalFunction1);
    auto & callExternalFunction2LambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.CallExternalFunction2);

    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    /*
     * Validate ReturnAddress function
     */
    {
      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.ReturnAddressFunction);
      AssertMemoryNodes(lambdaEntryNodes, {&returnAddressMallocMemoryNode});

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.ReturnAddressFunction);
      AssertMemoryNodes(lambdaExitNodes, {&returnAddressMallocMemoryNode});
    }

    /*
     * Validate CallExternalFunction1 function
     */
    {
      jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes({
        &returnAddressMallocMemoryNode,
        &callExternalFunction1MallocMemoryNode,
        &returnAddressLambdaMemoryNode,
        &callExternalFunction1LambdaMemoryNode,
        &callExternalFunction2LambdaMemoryNode,
        &externalMemoryNode});

      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.CallExternalFunction1);
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = provider.GetCallEntryNodes(*test.ExternalFunction1Call);
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = provider.GetCallExitNodes(*test.ExternalFunction1Call);
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.CallExternalFunction1);
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate CallExternalFunction2 function
     */
    {
      jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes({
        &returnAddressMallocMemoryNode,
        &callExternalFunction1MallocMemoryNode,
        &returnAddressLambdaMemoryNode,
        &callExternalFunction1LambdaMemoryNode,
        &callExternalFunction2LambdaMemoryNode,
        &externalMemoryNode});

      auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.CallExternalFunction2);
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = provider.GetCallEntryNodes(*test.ExternalFunction2Call);
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = provider.GetCallExitNodes(*test.ExternalFunction2Call);
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.CallExternalFunction2);
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }
  };

  EscapedMemoryTest2 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static void
TestEscapedMemory3()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](
    const EscapedMemoryTest3 & test,
    const jlm::aa::RegionAwareMemoryNodeProvider & provider,
    const jlm::aa::PointsToGraph & pointsToGraph)
  {
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto & deltaMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaGlobal);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedMemoryNodes({
      &lambdaMemoryNode,
      &deltaMemoryNode,
      &externalMemoryNode});

    auto & lambdaEntryNodes = provider.GetLambdaEntryNodes(*test.LambdaTest);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & callEntryNodes = provider.GetCallEntryNodes(*test.CallExternalFunction);
    AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

    auto & callExitNodes = provider.GetCallExitNodes(*test.CallExternalFunction);
    AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = provider.GetLambdaExitNodes(*test.LambdaTest);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  EscapedMemoryTest3 test;
//	jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto provider = jlm::aa::RegionAwareMemoryNodeProvider::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *provider, *pointsToGraph);
}

static int
TestRegionAwareMemoryNodeProvider()
{
  TestStore1();
  TestStore2();

  TestLoad1();
  TestLoad2();
  TestLoadFromUndef();

  TestCall1();
  TestCall2();

  TestIndirectCall();
  TestIndirectCall2();

  TestGamma();

  TestTheta();

  TestDelta1();
  TestDelta2();

  TestImports();

  TestPhi1();
  TestPhi2();

  TestEscapedMemory1();
  TestEscapedMemory2();
  TestEscapedMemory3();

  TestMemcpy();

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/TestRegionAwareMemoryNodeProvider", TestRegionAwareMemoryNodeProvider)