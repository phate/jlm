/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunSteensgaard(jlm::llvm::RvsdgModule & rvsdgModule)
{
  using namespace jlm::llvm;

  aa::Steensgaard steensgaard;
  jlm::util::StatisticsCollector statisticsCollector;
  return steensgaard.Analyze(rvsdgModule, statisticsCollector);
}

static void
AssertMemoryNodes(
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> &
        receivedMemoryNodes,
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> &
        expectedMemoryNodes)
{
  assert(receivedMemoryNodes == expectedMemoryNodes);
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
    auto & allocaAMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_a);
    auto & allocaBMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_b);
    auto & allocaCMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_c);
    auto & allocaDMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_d);

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes({
        &allocaAMemoryNode,
        &allocaBMemoryNode,
        &allocaCMemoryNode,
        &allocaDMemoryNode,
    });

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  jlm::tests::StoreTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & allocaAMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_a);
    auto & allocaBMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_b);
    auto & allocaPMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_p);
    auto & allocaXMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_x);
    auto & allocaYMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_y);

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &allocaAMemoryNode,
          &allocaBMemoryNode,
          &allocaPMemoryNode,
          &allocaXMemoryNode,
          &allocaYMemoryNode });

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  jlm::tests::StoreTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, { &lambdaMemoryNode, &externalMemoryNode });

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, { &lambdaMemoryNode, &externalMemoryNode });
  };

  jlm::tests::LoadTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & allocaAMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_a);
    auto & allocaBMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_b);
    auto & allocaPMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_p);
    auto & allocaXMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_x);
    auto & allocaYMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_y);

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &allocaAMemoryNode,
          &allocaBMemoryNode,
          &allocaPMemoryNode,
          &allocaXMemoryNode,
          &allocaYMemoryNode });

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  jlm::tests::LoadTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
                             const jlm::llvm::aa::PointsToGraph &)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.Lambda()).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.Lambda()).Size();

    assert(numLambdaEntryNodes == 0);
    assert(numLambdaExitNodes == 0);
  };

  jlm::tests::LoadFromUndefTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & allocaXMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_x);
    auto & allocaYMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_y);
    auto & allocaZMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_z);

    /*
     * Validate function f
     */
    {
      auto & lambdaFEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_f);
      AssertMemoryNodes(lambdaFEntryNodes, { &allocaXMemoryNode, &allocaYMemoryNode });

      auto & lambdaFExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_f);
      AssertMemoryNodes(lambdaFExitNodes, { &allocaXMemoryNode, &allocaYMemoryNode });
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaGEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_g);
      AssertMemoryNodes(lambdaGEntryNodes, { &allocaZMemoryNode });

      auto & lambdaGExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_g);
      AssertMemoryNodes(lambdaGExitNodes, { &allocaZMemoryNode });
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaHEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_h);
      AssertMemoryNodes(
          lambdaHEntryNodes,
          { &allocaXMemoryNode, &allocaYMemoryNode, &allocaZMemoryNode });

      auto & callFEntryNodes = modRefSummary.GetCallEntryNodes(test.CallF());
      AssertMemoryNodes(callFEntryNodes, { &allocaXMemoryNode, &allocaYMemoryNode });

      auto & callFExitNodes = modRefSummary.GetCallExitNodes(test.CallF());
      AssertMemoryNodes(callFExitNodes, { &allocaXMemoryNode, &allocaYMemoryNode });

      auto & callGEntryNodes = modRefSummary.GetCallEntryNodes(test.CallG());
      AssertMemoryNodes(callGEntryNodes, { &allocaZMemoryNode });

      auto & callGExitNodes = modRefSummary.GetCallExitNodes(test.CallG());
      AssertMemoryNodes(callGExitNodes, { &allocaZMemoryNode });

      auto & lambdaHExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_h);
      AssertMemoryNodes(
          lambdaHExitNodes,
          { &allocaXMemoryNode, &allocaYMemoryNode, &allocaZMemoryNode });
    }
  };

  jlm::tests::CallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & mallocMemoryNode = pointsToGraph.GetMallocNode(*test.malloc);

    /*
     * Validate function create
     */
    {
      auto & lambdaCreateEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_create);
      AssertMemoryNodes(lambdaCreateEntryNodes, { &mallocMemoryNode });

      auto & lambdaCreateExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_create);
      AssertMemoryNodes(lambdaCreateExitNodes, { &mallocMemoryNode });
    }

    /*
     * Validate function destroy
     */
    {
      auto & lambdaDestroyEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_destroy);
      AssertMemoryNodes(lambdaDestroyEntryNodes, { &mallocMemoryNode });

      auto & lambdaDestroyExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_destroy);
      AssertMemoryNodes(lambdaDestroyExitNodes, { &mallocMemoryNode });
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaTestEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaTestEntryNodes, { &mallocMemoryNode });

      auto & callCreate1EntryNodes = modRefSummary.GetCallEntryNodes(test.CallCreate1());
      AssertMemoryNodes(callCreate1EntryNodes, { &mallocMemoryNode });

      auto & callCreate1ExitNodes = modRefSummary.GetCallExitNodes(test.CallCreate1());
      AssertMemoryNodes(callCreate1ExitNodes, { &mallocMemoryNode });

      auto & callCreate2EntryNodes = modRefSummary.GetCallEntryNodes(test.CallCreate2());
      AssertMemoryNodes(callCreate2EntryNodes, { &mallocMemoryNode });

      auto & callCreate2ExitNodes = modRefSummary.GetCallExitNodes(test.CallCreate2());
      AssertMemoryNodes(callCreate2ExitNodes, { &mallocMemoryNode });

      auto & callDestroy1EntryNodes = modRefSummary.GetCallEntryNodes(test.CallDestroy1());
      AssertMemoryNodes(callDestroy1EntryNodes, { &mallocMemoryNode });

      auto & callDestroy1ExitNodes = modRefSummary.GetCallExitNodes(test.CallDestroy1());
      AssertMemoryNodes(callDestroy1ExitNodes, { &mallocMemoryNode });

      auto & callDestroy2EntryNodes = modRefSummary.GetCallEntryNodes(test.CallDestroy2());
      AssertMemoryNodes(callDestroy2EntryNodes, { &mallocMemoryNode });

      auto & callDestroy2ExitNodes = modRefSummary.GetCallExitNodes(test.CallDestroy2());
      AssertMemoryNodes(callDestroy2ExitNodes, { &mallocMemoryNode });

      auto & lambdaTestExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaTestExitNodes, { &mallocMemoryNode });
    }
  };

  jlm::tests::CallTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
                             [[maybe_unused]] const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function four
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaFour());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaFour());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function three
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaThree());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaThree());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function indcall
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaIndcall());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.CallIndcall());
      AssertMemoryNodes(callEntryNodes, {});

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.CallIndcall());
      AssertMemoryNodes(callExitNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaIndcall());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & callFourEntryNodes = modRefSummary.GetCallEntryNodes(test.CallFour());
      AssertMemoryNodes(callFourEntryNodes, {});

      auto & callFourExitNodes = modRefSummary.GetCallExitNodes(test.CallFour());
      AssertMemoryNodes(callFourExitNodes, {});

      auto & callThreeEntryNodes = modRefSummary.GetCallEntryNodes(test.CallThree());
      AssertMemoryNodes(callThreeEntryNodes, {});

      auto & callThreeExitNodes = modRefSummary.GetCallExitNodes(test.CallThree());
      AssertMemoryNodes(callThreeExitNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaExitNodes, {});
    }
  };

  jlm::tests::IndirectCallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

static void
TestIndirectCall2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::IndirectCallTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto & deltaG1MemoryNode = pointsToGraph.GetDeltaNode(test.GetDeltaG1());
    auto & deltaG2MemoryNode = pointsToGraph.GetDeltaNode(test.GetDeltaG2());

    auto & allocaPxMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPx());
    auto & allocaPyMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPy());
    auto & allocaPzMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPz());

    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> pY = {
      &allocaPyMemoryNode,
    };
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> pXZ = {
      &allocaPxMemoryNode,
      &allocaPzMemoryNode
    };
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> pXYZG1G2 = {
      &allocaPxMemoryNode,
      &allocaPyMemoryNode,
      &allocaPzMemoryNode,
      &deltaG1MemoryNode,
      &deltaG2MemoryNode
    };

    /*
     * Validate function four()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaFour());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaFour());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function three()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaThree());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaThree());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaI());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.GetIndirectCall());
      AssertMemoryNodes(callEntryNodes, {});

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.GetIndirectCall());
      AssertMemoryNodes(callExitNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaI());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function x()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaX());
      AssertMemoryNodes(lambdaEntryNodes, pXZ);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaX());
      AssertMemoryNodes(lambdaExitNodes, pXZ);
    }

    /*
     * Validate function y()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaY());
      AssertMemoryNodes(lambdaEntryNodes, pY);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaY());
      AssertMemoryNodes(lambdaExitNodes, pY);
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaEntryNodes, pXYZG1G2);

      auto & callXEntryNodes = modRefSummary.GetCallEntryNodes(test.GetTestCallX());
      AssertMemoryNodes(callXEntryNodes, pXZ);

      auto & callXExitNodes = modRefSummary.GetCallExitNodes(test.GetTestCallX());
      AssertMemoryNodes(callXExitNodes, pXZ);

      auto & callYEntryNodes = modRefSummary.GetCallEntryNodes(test.GetCallY());
      AssertMemoryNodes(callYEntryNodes, pY);

      auto & callYExitNodes = modRefSummary.GetCallExitNodes(test.GetCallY());
      AssertMemoryNodes(callYExitNodes, pY);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaExitNodes, pXYZG1G2);
    }

    /*
     * Validate function test2()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaTest2());
      AssertMemoryNodes(lambdaEntryNodes, pXZ);

      auto & callXEntryNodes = modRefSummary.GetCallEntryNodes(test.GetTest2CallX());
      AssertMemoryNodes(callXEntryNodes, pXZ);

      auto & callXExitNodes = modRefSummary.GetCallExitNodes(test.GetTest2CallX());
      AssertMemoryNodes(callXExitNodes, pXZ);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaTest2());
      AssertMemoryNodes(lambdaExitNodes, pXZ);
    }
  };

  jlm::tests::IndirectCallTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, { &lambdaMemoryNode, &externalMemoryNode });

    auto gammaEntryNodes = modRefSummary.GetGammaEntryNodes(*test.gamma);
    AssertMemoryNodes(gammaEntryNodes, {});

    auto gammaExitNodes = modRefSummary.GetGammaExitNodes(*test.gamma);
    AssertMemoryNodes(gammaExitNodes, {});

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, { &lambdaMemoryNode, &externalMemoryNode });
  };

  jlm::tests::GammaTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda);
    AssertMemoryNodes(lambdaEntryNodes, { &lambdaMemoryNode, &externalMemoryNode });

    auto & thetaEntryExitNodes = modRefSummary.GetThetaEntryExitNodes(*test.theta);
    AssertMemoryNodes(thetaEntryExitNodes, { &lambdaMemoryNode, &externalMemoryNode });

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda);
    AssertMemoryNodes(lambdaExitNodes, { &lambdaMemoryNode, &externalMemoryNode });
  };

  jlm::tests::ThetaTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & deltaFNode = pointsToGraph.GetDeltaNode(*test.delta_f);

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_g);
      AssertMemoryNodes(lambdaEntryNodes, { &deltaFNode });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_g);
      AssertMemoryNodes(lambdaExitNodes, { &deltaFNode });
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_h);
      AssertMemoryNodes(lambdaEntryNodes, { &deltaFNode });

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.CallG());
      AssertMemoryNodes(callEntryNodes, { &deltaFNode });

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.CallG());
      AssertMemoryNodes(callExitNodes, { &deltaFNode });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_h);
      AssertMemoryNodes(lambdaExitNodes, { &deltaFNode });
    }
  };

  jlm::tests::DeltaTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & deltaD1Node = pointsToGraph.GetDeltaNode(*test.delta_d1);
    auto & deltaD2Node = pointsToGraph.GetDeltaNode(*test.delta_d2);

    /*
     * Validate function f1
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_f1);
      AssertMemoryNodes(lambdaEntryNodes, { &deltaD1Node });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_f1);
      AssertMemoryNodes(lambdaExitNodes, { &deltaD1Node });
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_f2);
      AssertMemoryNodes(lambdaEntryNodes, { &deltaD1Node, &deltaD2Node });

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.CallF1());
      AssertMemoryNodes(callEntryNodes, { &deltaD1Node });

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.CallF1());
      AssertMemoryNodes(callExitNodes, { &deltaD1Node });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_f2);
      AssertMemoryNodes(lambdaExitNodes, { &deltaD1Node, &deltaD2Node });
    }
  };

  jlm::tests::DeltaTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & importD1Node = pointsToGraph.GetImportNode(*test.import_d1);
    auto & importD2Node = pointsToGraph.GetImportNode(*test.import_d2);

    /*
     * Validate function f1
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_f1);
      AssertMemoryNodes(lambdaEntryNodes, { &importD1Node });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_f1);
      AssertMemoryNodes(lambdaExitNodes, { &importD1Node });
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_f2);
      AssertMemoryNodes(lambdaEntryNodes, { &importD1Node, &importD2Node });

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.CallF1());
      AssertMemoryNodes(callEntryNodes, { &importD1Node });

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.CallF1());
      AssertMemoryNodes(callExitNodes, { &importD1Node });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_f2);
      AssertMemoryNodes(lambdaExitNodes, { &importD1Node, &importD2Node });
    }
  };

  jlm::tests::ImportTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

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
    auto & resultAllocaNode = pointsToGraph.GetAllocaNode(*test.alloca);

    /*
     * Validate function fib
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_fib);
      AssertMemoryNodes(lambdaEntryNodes, { &resultAllocaNode });

      auto & callFibM1EntryNodes = modRefSummary.GetCallEntryNodes(test.CallFibm1());
      AssertMemoryNodes(callFibM1EntryNodes, { &resultAllocaNode });

      auto & callFibM1ExitNodes = modRefSummary.GetCallExitNodes(test.CallFibm1());
      AssertMemoryNodes(callFibM1ExitNodes, { &resultAllocaNode });

      auto & callFibM2EntryNodes = modRefSummary.GetCallEntryNodes(test.CallFibm2());
      AssertMemoryNodes(callFibM2EntryNodes, { &resultAllocaNode });

      auto & callFibM2ExitNodes = modRefSummary.GetCallExitNodes(test.CallFibm2());
      AssertMemoryNodes(callFibM2ExitNodes, { &resultAllocaNode });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_fib);
      AssertMemoryNodes(lambdaExitNodes, { &resultAllocaNode });
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaEntryNodes, { &resultAllocaNode });

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.CallFib());
      AssertMemoryNodes(callEntryNodes, { &resultAllocaNode });

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.CallFib());
      AssertMemoryNodes(callExitNodes, { &resultAllocaNode });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.lambda_test);
      AssertMemoryNodes(lambdaExitNodes, { &resultAllocaNode });
    }
  };

  jlm::tests::PhiTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

static void
TestPhi2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::PhiTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto & pTestAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPTestAlloca());
    auto & paAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPaAlloca());
    auto & pbAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPbAlloca());
    auto & pcAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPcAlloca());
    auto & pdAllocaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPdAlloca());

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &pTestAllocaMemoryNode,
          &paAllocaMemoryNode,
          &pbAllocaMemoryNode,
          &pcAllocaMemoryNode,
          &pdAllocaMemoryNode });

    /*
     * Validate function eight()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaEight());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaEight());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaI());
      AssertMemoryNodes(lambdaEntryNodes, {});

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.GetIndirectCall());
      AssertMemoryNodes(callEntryNodes, {});

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.GetIndirectCall());
      AssertMemoryNodes(callExitNodes, {});

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaI());
      AssertMemoryNodes(lambdaExitNodes, {});
    }

    /*
     * Validate function a()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaA());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callBEntryNodes = modRefSummary.GetCallEntryNodes(test.GetCallB());
      AssertMemoryNodes(callBEntryNodes, expectedMemoryNodes);

      auto & callBExitNodes = modRefSummary.GetCallExitNodes(test.GetCallB());
      AssertMemoryNodes(callBExitNodes, expectedMemoryNodes);

      auto & callDEntryNodes = modRefSummary.GetCallEntryNodes(test.GetCallD());
      AssertMemoryNodes(callDEntryNodes, expectedMemoryNodes);

      auto & callDExitNodes = modRefSummary.GetCallExitNodes(test.GetCallD());
      AssertMemoryNodes(callDExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaA());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function b()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaB());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callIEntryNodes = modRefSummary.GetCallEntryNodes(test.GetCallI());
      AssertMemoryNodes(callIEntryNodes, {});

      auto & callIExitNodes = modRefSummary.GetCallExitNodes(test.GetCallI());
      AssertMemoryNodes(callIExitNodes, {});

      auto & callCEntryNodes = modRefSummary.GetCallEntryNodes(test.GetCallC());
      AssertMemoryNodes(callCEntryNodes, expectedMemoryNodes);

      auto & callCExitNodes = modRefSummary.GetCallExitNodes(test.GetCallC());
      AssertMemoryNodes(callCExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaB());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function c()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaC());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.GetCallAFromC());
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.GetCallAFromC());
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaC());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function d()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaD());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.GetCallAFromD());
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.GetCallAFromD());
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaD());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.GetCallAFromTest());
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.GetCallAFromTest());
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.GetLambdaTest());
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }
  };

  jlm::tests::PhiTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

static void
TestPhiWithDelta()
{
  // Assert
  jlm::tests::PhiWithDeltaTest test;
  std::unordered_map<const jlm::rvsdg::Output *, std::string> outputMap;
  std::cout << jlm::rvsdg::view(&test.graph().GetRootRegion(), outputMap) << std::flush;

  auto pointsToGraph = RunSteensgaard(test.module());
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph, outputMap) << std::flush;

  // Act
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  // Assert
  // Nothing needs to be validated as there are only phi and delta nodes in the RVSDG.
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
    auto & localArrayMemoryNode = pointsToGraph.GetDeltaNode(test.LocalArray());
    auto & globalArrayMemoryNode = pointsToGraph.GetDeltaNode(test.GlobalArray());

    /*
     * Validate function f
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.LambdaF());
      AssertMemoryNodes(lambdaEntryNodes, { &globalArrayMemoryNode, &localArrayMemoryNode });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.LambdaF());
      AssertMemoryNodes(lambdaExitNodes, { &globalArrayMemoryNode, &localArrayMemoryNode });
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(test.LambdaG());
      AssertMemoryNodes(lambdaEntryNodes, { &localArrayMemoryNode, &globalArrayMemoryNode });

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(test.CallF());
      AssertMemoryNodes(callEntryNodes, { &globalArrayMemoryNode, &localArrayMemoryNode });

      auto & callExitNodes = modRefSummary.GetCallExitNodes(test.CallF());
      AssertMemoryNodes(callExitNodes, { &globalArrayMemoryNode, &localArrayMemoryNode });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(test.LambdaG());
      AssertMemoryNodes(lambdaExitNodes, { &localArrayMemoryNode, &globalArrayMemoryNode });
    }
  };

  jlm::tests::MemcpyTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

static void
TestEscapedMemory1()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::EscapedMemoryTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto & deltaAMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaA);
    auto & deltaBMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaB);
    auto & deltaXMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaX);
    auto & deltaYMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaY);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaMemoryNode,
          &deltaAMemoryNode,
          &deltaBMemoryNode,
          &deltaXMemoryNode,
          &deltaYMemoryNode,
          &externalMemoryNode });

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.LambdaTest);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.LambdaTest);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  jlm::tests::EscapedMemoryTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

static void
TestEscapedMemory2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::EscapedMemoryTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto & returnAddressMallocMemoryNode = pointsToGraph.GetMallocNode(*test.ReturnAddressMalloc);
    auto & callExternalFunction1MallocMemoryNode =
        pointsToGraph.GetMallocNode(*test.CallExternalFunction1Malloc);

    auto & returnAddressLambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.ReturnAddressFunction);
    auto & callExternalFunction1LambdaMemoryNode =
        pointsToGraph.GetLambdaNode(*test.CallExternalFunction1);
    auto & callExternalFunction2LambdaMemoryNode =
        pointsToGraph.GetLambdaNode(*test.CallExternalFunction2);

    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    /*
     * Validate ReturnAddress function
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.ReturnAddressFunction);
      AssertMemoryNodes(lambdaEntryNodes, { &returnAddressMallocMemoryNode });

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.ReturnAddressFunction);
      AssertMemoryNodes(lambdaExitNodes, { &returnAddressMallocMemoryNode });
    }

    /*
     * Validate CallExternalFunction1 function
     */
    {
      jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
          { &returnAddressMallocMemoryNode,
            &callExternalFunction1MallocMemoryNode,
            &returnAddressLambdaMemoryNode,
            &callExternalFunction1LambdaMemoryNode,
            &callExternalFunction2LambdaMemoryNode,
            &externalMemoryNode });

      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.CallExternalFunction1);
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(*test.ExternalFunction1Call);
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = modRefSummary.GetCallExitNodes(*test.ExternalFunction1Call);
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.CallExternalFunction1);
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }

    /*
     * Validate CallExternalFunction2 function
     */
    {
      jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
          { &returnAddressMallocMemoryNode,
            &callExternalFunction1MallocMemoryNode,
            &returnAddressLambdaMemoryNode,
            &callExternalFunction1LambdaMemoryNode,
            &callExternalFunction2LambdaMemoryNode,
            &externalMemoryNode });

      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.CallExternalFunction2);
      AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

      auto & callEntryNodes = modRefSummary.GetCallEntryNodes(*test.ExternalFunction2Call);
      AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

      auto & callExitNodes = modRefSummary.GetCallExitNodes(*test.ExternalFunction2Call);
      AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.CallExternalFunction2);
      AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
    }
  };

  jlm::tests::EscapedMemoryTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

static void
TestEscapedMemory3()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::EscapedMemoryTest3 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto & deltaMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaGlobal);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaMemoryNode, &deltaMemoryNode, &externalMemoryNode });

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryNodes(*test.LambdaTest);
    AssertMemoryNodes(lambdaEntryNodes, expectedMemoryNodes);

    auto & callEntryNodes = modRefSummary.GetCallEntryNodes(*test.CallExternalFunction);
    AssertMemoryNodes(callEntryNodes, expectedMemoryNodes);

    auto & callExitNodes = modRefSummary.GetCallExitNodes(*test.CallExternalFunction);
    AssertMemoryNodes(callExitNodes, expectedMemoryNodes);

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitNodes(*test.LambdaTest);
    AssertMemoryNodes(lambdaExitNodes, expectedMemoryNodes);
  };

  jlm::tests::EscapedMemoryTest3 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}

static void
TestStatistics()
{
  using namespace jlm;

  // Arrange
  tests::LoadTest1 test;
  auto pointsToGraph = RunSteensgaard(test.module());

  util::StatisticsCollectorSettings statisticsCollectorSettings(
      { util::Statistics::Id::RegionAwareModRefSummarizer });
  util::StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  // Act
  jlm::llvm::aa::RegionAwareModRefSummarizer::Create(
      test.module(),
      *pointsToGraph,
      statisticsCollector);

  // Assert
  assert(statisticsCollector.NumCollectedStatistics() == 1);
  auto & statistics = *statisticsCollector.CollectedStatistics().begin();

  assert(statistics.GetMeasurementValue<uint64_t>("#RvsdgNodes") == 3);
  assert(statistics.GetMeasurementValue<uint64_t>("#RvsdgRegions") == 2);
  assert(statistics.GetMeasurementValue<uint64_t>("#PointsToGraphMemoryNodes") == 2);
  assert(statistics.GetMeasurementValue<uint64_t>("#CallGraphSccs") == 2);

  assert(statistics.HasTimer("CallGraphTimer"));
  assert(statistics.HasTimer("AnnotationTimer"));
  assert(statistics.HasTimer("PropagateTimer"));
}

static void
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
  TestPhiWithDelta();

  TestEscapedMemory1();
  TestEscapedMemory2();
  TestEscapedMemory3();

  TestMemcpy();

  TestStatistics();
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests",
    TestRegionAwareMemoryNodeProvider)
