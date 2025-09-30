/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunAndersen(jlm::llvm::RvsdgModule & rvsdgModule)
{
  jlm::llvm::aa::Andersen andersen;
  return andersen.Analyze(rvsdgModule);
}

// Helper for comparing HashSets of MemoryNodes without needing explicit constructors
static bool
setsEqual(
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> &
        receivedMemoryNodes,
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> &
        expectedMemoryNodes)
{
  return receivedMemoryNodes == expectedMemoryNodes;
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

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        {
            &allocaAMemoryNode,
            &allocaBMemoryNode,
            &allocaCMemoryNode,
            &allocaDMemoryNode,
        });

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
  };

  jlm::tests::StoreTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestStore1",
    TestStore1)

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

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
  };

  jlm::tests::StoreTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestStore2",
    TestStore2)

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

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    assert(setsEqual(lambdaEntryNodes, { &lambdaMemoryNode, &externalMemoryNode }));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    assert(setsEqual(lambdaExitNodes, { &lambdaMemoryNode, &externalMemoryNode }));
  };

  jlm::tests::LoadTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestLoad1",
    TestLoad1)

static void
TestLoad2()
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::tests::LoadTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary)
  {
    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    assert(setsEqual(lambdaEntryNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    assert(setsEqual(lambdaExitNodes, {}));
  };

  jlm::tests::LoadTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Act
   */
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary);
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestLoad2",
    TestLoad2)

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
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.Lambda()).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.Lambda()).Size();

    assert(numLambdaEntryNodes == 0);
    assert(numLambdaExitNodes == 0);
  };

  jlm::tests::LoadFromUndefTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestLoadFromUndef",
    TestLoadFromUndef)

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
      auto & lambdaFEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f);
      assert(setsEqual(lambdaFEntryNodes, { &allocaXMemoryNode, &allocaYMemoryNode }));

      auto & lambdaFExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f);
      assert(setsEqual(lambdaFExitNodes, { &allocaXMemoryNode, &allocaYMemoryNode }));
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaGEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g);
      assert(setsEqual(lambdaGEntryNodes, { &allocaZMemoryNode }));

      auto & lambdaGExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g);
      assert(setsEqual(lambdaGExitNodes, { &allocaZMemoryNode }));
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaHEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h);
      assert(setsEqual(lambdaHEntryNodes, {}));

      auto & callFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF());
      assert(setsEqual(callFNodes, { &allocaXMemoryNode, &allocaYMemoryNode }));

      auto & callGNodes = modRefSummary.GetSimpleNodeModRef(test.CallG());
      assert(setsEqual(callGNodes, { &allocaZMemoryNode }));

      auto & lambdaHExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h);
      assert(setsEqual(lambdaHExitNodes, {}));
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
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestCall1",
    TestCall1)

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
      auto & lambdaCreateEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_create);
      assert(setsEqual(lambdaCreateEntryNodes, { &mallocMemoryNode }));

      auto & lambdaCreateExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_create);
      assert(setsEqual(lambdaCreateExitNodes, { &mallocMemoryNode }));
    }

    /*
     * Validate function destroy
     */
    {
      auto & lambdaDestroyEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_destroy);
      assert(setsEqual(lambdaDestroyEntryNodes, { &mallocMemoryNode }));

      auto & lambdaDestroyExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_destroy);
      assert(setsEqual(lambdaDestroyExitNodes, { &mallocMemoryNode }));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaTestEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test);
      assert(setsEqual(lambdaTestEntryNodes, { &mallocMemoryNode }));

      auto & callCreate1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate1());
      assert(setsEqual(callCreate1Nodes, { &mallocMemoryNode }));

      auto & callCreate2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate2());
      assert(setsEqual(callCreate2Nodes, { &mallocMemoryNode }));

      auto & callDestroy1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy1());
      assert(setsEqual(callDestroy1Nodes, { &mallocMemoryNode }));

      auto & callDestroy2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy2());
      assert(setsEqual(callDestroy2Nodes, { &mallocMemoryNode }));

      auto & lambdaTestExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test);
      assert(setsEqual(lambdaTestExitNodes, { &mallocMemoryNode }));
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
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestCall2",
    TestCall2)

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
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaFour());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaFour());
      assert(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function three
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree());
      assert(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function indcall
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaIndcall());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallIndcall());
      assert(setsEqual(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaIndcall());
      assert(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & callFourNodes = modRefSummary.GetSimpleNodeModRef(test.CallFour());
      assert(setsEqual(callFourNodes, {}));

      auto & callThreeNodes = modRefSummary.GetSimpleNodeModRef(test.CallThree());
      assert(setsEqual(callThreeNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      assert(setsEqual(lambdaExitNodes, {}));
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
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestIndirectCall",
    TestIndirectCall)

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

    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> pX = {
      &allocaPxMemoryNode,
    };
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> pY = {
      &allocaPyMemoryNode,
    };
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> pZ = {
      &allocaPzMemoryNode,
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
    const jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> pG1G2 = {
      &deltaG1MemoryNode,
      &deltaG2MemoryNode
    };

    /*
     * Validate function four()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaFour());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaFour());
      assert(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function three()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree());
      assert(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaI());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetIndirectCall());
      assert(setsEqual(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaI());
      assert(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function x()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaX());
      assert(setsEqual(lambdaEntryNodes, pXZ));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaX());
      assert(setsEqual(lambdaExitNodes, pXZ));
    }

    /*
     * Validate function y()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaY());
      assert(setsEqual(lambdaEntryNodes, pY));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaY());
      assert(setsEqual(lambdaExitNodes, pY));
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      assert(setsEqual(lambdaEntryNodes, pG1G2));

      auto & callXNodes = modRefSummary.GetSimpleNodeModRef(test.GetTestCallX());
      assert(setsEqual(callXNodes, pX));

      auto & callYNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallY());
      assert(setsEqual(callYNodes, pY));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      assert(setsEqual(lambdaExitNodes, pG1G2));
    }

    /*
     * Validate function test2()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest2());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & callXNodes = modRefSummary.GetSimpleNodeModRef(test.GetTest2CallX());
      assert(setsEqual(callXNodes, pZ));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest2());
      assert(setsEqual(lambdaExitNodes, {}));
    }
  };

  jlm::tests::IndirectCallTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestIndirectCall2",
    TestIndirectCall2)

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

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    assert(setsEqual(lambdaEntryNodes, { &lambdaMemoryNode, &externalMemoryNode }));

    auto gammaEntryNodes = modRefSummary.GetGammaEntryModRef(*test.gamma);
    assert(setsEqual(gammaEntryNodes, {}));

    auto gammaExitNodes = modRefSummary.GetGammaExitModRef(*test.gamma);
    assert(setsEqual(gammaExitNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    assert(setsEqual(lambdaExitNodes, { &lambdaMemoryNode, &externalMemoryNode }));
  };

  jlm::tests::GammaTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestGamma",
    TestGamma)

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

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    assert(setsEqual(lambdaEntryNodes, { &lambdaMemoryNode, &externalMemoryNode }));

    auto & thetaEntryExitNodes = modRefSummary.GetThetaModRef(*test.theta);
    assert(setsEqual(thetaEntryExitNodes, { &lambdaMemoryNode, &externalMemoryNode }));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    assert(setsEqual(lambdaExitNodes, { &lambdaMemoryNode, &externalMemoryNode }));
  };

  jlm::tests::ThetaTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestTheta",
    TestTheta)

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
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g);
      assert(setsEqual(lambdaEntryNodes, { &deltaFNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g);
      assert(setsEqual(lambdaExitNodes, { &deltaFNode }));
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h);
      assert(setsEqual(lambdaEntryNodes, { &deltaFNode }));

      auto & callEntryNodes = modRefSummary.GetSimpleNodeModRef(test.CallG());
      assert(setsEqual(callEntryNodes, { &deltaFNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h);
      assert(setsEqual(lambdaExitNodes, { &deltaFNode }));
    }
  };

  jlm::tests::DeltaTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestDelta1",
    TestDelta1)

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
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f1);
      assert(setsEqual(lambdaEntryNodes, { &deltaD1Node }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1);
      assert(setsEqual(lambdaExitNodes, { &deltaD1Node }));
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2);
      assert(setsEqual(lambdaEntryNodes, { &deltaD1Node, &deltaD2Node }));

      auto & callEntryNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1());
      assert(setsEqual(callEntryNodes, { &deltaD1Node }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2);
      assert(setsEqual(lambdaExitNodes, { &deltaD1Node, &deltaD2Node }));
    }
  };

  jlm::tests::DeltaTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestDelta2",
    TestDelta2)

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
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f1);
      assert(setsEqual(lambdaEntryNodes, { &importD1Node }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1);
      assert(setsEqual(lambdaExitNodes, { &importD1Node }));
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2);
      assert(setsEqual(lambdaEntryNodes, { &importD1Node, &importD2Node }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1());
      assert(setsEqual(callNodes, { &importD1Node }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2);
      assert(setsEqual(lambdaExitNodes, { &importD1Node, &importD2Node }));
    }
  };

  jlm::tests::ImportTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestImports",
    TestImports)

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
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_fib);
      assert(setsEqual(lambdaEntryNodes, { &resultAllocaNode }));

      auto & callFibM1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallFibm1());
      assert(setsEqual(callFibM1Nodes, { &resultAllocaNode }));

      auto & callFibM2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallFibm2());
      assert(setsEqual(callFibM2Nodes, { &resultAllocaNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_fib);
      assert(setsEqual(lambdaExitNodes, { &resultAllocaNode }));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test);
      assert(setsEqual(lambdaEntryNodes, { &resultAllocaNode }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallFib());
      assert(setsEqual(callNodes, { &resultAllocaNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test);
      assert(setsEqual(lambdaExitNodes, { &resultAllocaNode }));
    }
  };

  jlm::tests::PhiTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestPhi1",
    TestPhi1)

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
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaEight());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaEight());
      assert(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaI());
      assert(setsEqual(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetIndirectCall());
      assert(setsEqual(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaI());
      assert(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function a()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaA());
      assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callBNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallB());
      assert(setsEqual(callBNodes, expectedMemoryNodes));

      auto & callDNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallD());
      assert(setsEqual(callDNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaA());
      assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }

    /*
     * Validate function b()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaB());
      assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callINodes = modRefSummary.GetSimpleNodeModRef(test.GetCallI());
      assert(setsEqual(callINodes, {}));

      auto & callCNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallC());
      assert(setsEqual(callCNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaB());
      assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }

    /*
     * Validate function c()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaC());
      assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromC());
      assert(setsEqual(callNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaC());
      assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }

    /*
     * Validate function d()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaD());
      assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromD());
      assert(setsEqual(callNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaD());
      assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromTest());
      assert(setsEqual(callNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }
  };

  jlm::tests::PhiTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestPhi2",
    TestPhi2)

static void
TestPhiWithDelta()
{
  // Assert
  jlm::tests::PhiWithDeltaTest test;
  std::unordered_map<const jlm::rvsdg::Output *, std::string> outputMap;
  std::cout << jlm::rvsdg::view(&test.graph().GetRootRegion(), outputMap) << std::flush;

  auto pointsToGraph = RunAndersen(test.module());
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph, outputMap) << std::flush;

  // Act
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  // Assert
  // Nothing needs to be validated as there are only phi and delta nodes in the RVSDG.
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestPhiWithDelta",
    TestPhiWithDelta)

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
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaF());
      assert(setsEqual(lambdaEntryNodes, { &globalArrayMemoryNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaF());
      assert(setsEqual(lambdaExitNodes, { &globalArrayMemoryNode }));
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaG());
      assert(setsEqual(lambdaEntryNodes, { &localArrayMemoryNode, &globalArrayMemoryNode }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallF());
      assert(setsEqual(callNodes, { &globalArrayMemoryNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaG());
      assert(setsEqual(lambdaExitNodes, { &localArrayMemoryNode, &globalArrayMemoryNode }));
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
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  /*
   * Assert
   */
  ValidateProvider(test, *modRefSummary, *pointsToGraph);
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestMemcpy",
    TestMemcpy)

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

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.LambdaTest);
    assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.LambdaTest);
    assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
  };

  jlm::tests::EscapedMemoryTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestEscapedMemory1",
    TestEscapedMemory1)

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
    auto & externalFunction1ImportMemoryNode = pointsToGraph.GetImportNode(*test.ExternalFunction1Import);
    auto & externalFunction2ImportMemoryNode = pointsToGraph.GetImportNode(*test.ExternalFunction2Import);

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
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.ReturnAddressFunction);
      assert(setsEqual(lambdaEntryNodes, { &returnAddressMallocMemoryNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.ReturnAddressFunction);
      assert(setsEqual(lambdaExitNodes, { &returnAddressMallocMemoryNode }));
    }

    /*
     * Validate CallExternalFunction1 function
     */
    {
      jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
          {
            &externalFunction1ImportMemoryNode,
            &externalFunction2ImportMemoryNode,
            &returnAddressMallocMemoryNode,
            &callExternalFunction1MallocMemoryNode,
            &returnAddressLambdaMemoryNode,
            &callExternalFunction1LambdaMemoryNode,
            &callExternalFunction2LambdaMemoryNode,
            &externalMemoryNode });

      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.CallExternalFunction1);
      assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.ExternalFunction1Call);
      assert(setsEqual(callNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.CallExternalFunction1);
      assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }

    /*
     * Validate CallExternalFunction2 function
     */
    {
      jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
          {
            &externalFunction1ImportMemoryNode,
            &externalFunction2ImportMemoryNode,
            &returnAddressMallocMemoryNode,
            &callExternalFunction1MallocMemoryNode,
            &returnAddressLambdaMemoryNode,
            &callExternalFunction1LambdaMemoryNode,
            &callExternalFunction2LambdaMemoryNode,
            &externalMemoryNode });

      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.CallExternalFunction2);
      assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.ExternalFunction2Call);
      assert(setsEqual(callNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.CallExternalFunction2);
      assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }
  };

  jlm::tests::EscapedMemoryTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestEscapedMemory2",
    TestEscapedMemory2)

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
    auto & importedFunctionMemoryNode = pointsToGraph.GetImportNode(*test.ImportExternalFunction);
    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto & deltaMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaGlobal);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &importedFunctionMemoryNode, &lambdaMemoryNode, &deltaMemoryNode, &externalMemoryNode });

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.LambdaTest);
    assert(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

    auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.CallExternalFunction);
    assert(setsEqual(callNodes, expectedMemoryNodes));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.LambdaTest);
    assert(setsEqual(lambdaExitNodes, expectedMemoryNodes));
  };

  jlm::tests::EscapedMemoryTest3 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestEscapedMemory3",
    TestEscapedMemory3)

static void
TestStatistics()
{
  using namespace jlm;

  // Arrange
  tests::LoadTest2 test;
  auto pointsToGraph = RunAndersen(test.module());

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

  assert(statistics.GetMeasurementValue<uint64_t>("#RvsdgNodes") == 6);
  assert(statistics.GetMeasurementValue<uint64_t>("#RvsdgRegions") == 2);
  assert(statistics.GetMeasurementValue<uint64_t>("#PointsToGraphMemoryNodes") == 7);
  assert(statistics.GetMeasurementValue<uint64_t>("#SimpleAllocas") == 5);
  assert(statistics.GetMeasurementValue<uint64_t>("#NonReentrantAllocas") == 5);
  assert(statistics.GetMeasurementValue<uint64_t>("#CallGraphSccs") == 1);

  assert(statistics.HasTimer("SimpleAllocasSetTimer"));
  assert(statistics.HasTimer("NonReentrantAllocaSetsTimer"));
  assert(statistics.HasTimer("CallGraphTimer"));
  assert(statistics.HasTimer("AllocasDeadInSccsTimer"));
  assert(statistics.HasTimer("CreateExternalModRefSetTimer"));
  assert(statistics.HasTimer("AnnotationTimer"));
  assert(statistics.HasTimer("SolvingTimer"));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizerTests-TestStatistics",
    TestStatistics)
