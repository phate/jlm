/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/DotWriter.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/UnitType.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunAndersen(jlm::llvm::LlvmRvsdgModule & rvsdgModule)
{
  jlm::llvm::aa::Andersen andersen;
  return andersen.Analyze(rvsdgModule);
}

// Helper for comparing HashSets of MemoryNodes without needing explicit constructors
static bool
setsEqual(
    const jlm::util::HashSet<jlm::llvm::aa::PointsToGraph::NodeIndex> & receivedMemoryNodes,
    const jlm::util::HashSet<jlm::llvm::aa::PointsToGraph::NodeIndex> & expectedMemoryNodes)
{
  return receivedMemoryNodes == expectedMemoryNodes;
}

TEST(RegionAwareModRefSummarizerTests, TestStore1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::StoreTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto allocaAMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_a);

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));

    auto storeANode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(test.alloca_a->output(0)->SingleUser());
    EXPECT_TRUE(jlm::rvsdg::is<jlm::llvm::StoreNonVolatileOperation>(storeANode));

    auto & storeANodes = modRefSummary.GetSimpleNodeModRef(*storeANode);
    EXPECT_TRUE(setsEqual(storeANodes, { allocaAMemoryNode }));
  };

  jlm::llvm::StoreTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*PointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestStore2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::StoreTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto allocaAMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_a);
    auto allocaBMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_b);
    auto allocaPMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_p);
    auto allocaXMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_x);
    auto allocaYMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_y);

    jlm::util::HashSet expectedMemoryNodes{ allocaAMemoryNode,
                                            allocaBMemoryNode,
                                            allocaPMemoryNode,
                                            allocaXMemoryNode,
                                            allocaYMemoryNode };

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
  };

  jlm::llvm::StoreTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*PointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestLoad1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::LoadTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto externalMemoryNode = pointsToGraph.getExternalMemoryNode();

    auto lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaEntryNodes, { externalMemoryNode }));

    auto lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaExitNodes, { externalMemoryNode }));
  };

  jlm::llvm::LoadTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestLoad2)
{
  /*
   * Arrange
   */
  auto ValidateProvider =
      [](const jlm::llvm::LoadTest2 & test, const jlm::llvm::aa::ModRefSummary & modRefSummary)
  {
    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
  };

  jlm::llvm::LoadTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestLoadFromUndef)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::LoadFromUndefTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph &)
  {
    auto numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.Lambda()).Size();
    auto numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.Lambda()).Size();

    EXPECT_EQ(numLambdaEntryNodes, 0u);
    EXPECT_EQ(numLambdaExitNodes, 0u);
  };

  jlm::llvm::LoadFromUndefTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestCall1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::CallTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto allocaXMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_x);
    auto allocaYMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_y);
    auto allocaZMemoryNode = pointsToGraph.getNodeForAlloca(*test.alloca_z);

    /*
     * Validate function f
     */
    {
      auto & lambdaFEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f);
      EXPECT_TRUE(setsEqual(lambdaFEntryNodes, { allocaXMemoryNode, allocaYMemoryNode }));

      auto & lambdaFExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f);
      EXPECT_TRUE(setsEqual(lambdaFExitNodes, { allocaXMemoryNode, allocaYMemoryNode }));
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaGEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g);
      EXPECT_TRUE(setsEqual(lambdaGEntryNodes, { allocaZMemoryNode }));

      auto & lambdaGExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g);
      EXPECT_TRUE(setsEqual(lambdaGExitNodes, { allocaZMemoryNode }));
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaHEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h);
      EXPECT_TRUE(setsEqual(lambdaHEntryNodes, {}));

      auto & callFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF());
      EXPECT_TRUE(setsEqual(callFNodes, { allocaXMemoryNode, allocaYMemoryNode }));

      auto & callGNodes = modRefSummary.GetSimpleNodeModRef(test.CallG());
      EXPECT_TRUE(setsEqual(callGNodes, { allocaZMemoryNode }));

      auto & lambdaHExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h);
      EXPECT_TRUE(setsEqual(lambdaHExitNodes, {}));
    }
  };

  jlm::llvm::CallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*PointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestCall2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::CallTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto mallocMemoryNode = pointsToGraph.getNodeForMalloc(*test.malloc);

    /*
     * Validate function create
     */
    {
      auto & lambdaCreateEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_create);
      EXPECT_TRUE(setsEqual(lambdaCreateEntryNodes, { mallocMemoryNode }));

      auto & lambdaCreateExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_create);
      EXPECT_TRUE(setsEqual(lambdaCreateExitNodes, { mallocMemoryNode }));
    }

    /*
     * Validate function destroy
     */
    {
      auto & lambdaDestroyEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_destroy);
      EXPECT_TRUE(setsEqual(lambdaDestroyEntryNodes, { mallocMemoryNode }));

      auto & lambdaDestroyExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_destroy);
      EXPECT_TRUE(setsEqual(lambdaDestroyExitNodes, { mallocMemoryNode }));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaTestEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test);
      EXPECT_TRUE(setsEqual(lambdaTestEntryNodes, { mallocMemoryNode }));

      auto & callCreate1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate1());
      EXPECT_TRUE(setsEqual(callCreate1Nodes, { mallocMemoryNode }));

      auto & callCreate2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate2());
      EXPECT_TRUE(setsEqual(callCreate2Nodes, { mallocMemoryNode }));

      auto & callDestroy1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy1());
      EXPECT_TRUE(setsEqual(callDestroy1Nodes, { mallocMemoryNode }));

      auto & callDestroy2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy2());
      EXPECT_TRUE(setsEqual(callDestroy2Nodes, { mallocMemoryNode }));

      auto & lambdaTestExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test);
      EXPECT_TRUE(setsEqual(lambdaTestExitNodes, { mallocMemoryNode }));
    }
  };

  jlm::llvm::CallTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*PointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestIndirectCall)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::IndirectCallTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             [[maybe_unused]] const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    /*
     * Validate function four
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaFour());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaFour());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function three
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function indcall
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaIndcall());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallIndcall());
      EXPECT_TRUE(setsEqual(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaIndcall());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & callFourNodes = modRefSummary.GetSimpleNodeModRef(test.CallFour());
      EXPECT_TRUE(setsEqual(callFourNodes, {}));

      auto & callThreeNodes = modRefSummary.GetSimpleNodeModRef(test.CallThree());
      EXPECT_TRUE(setsEqual(callThreeNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }
  };

  jlm::llvm::IndirectCallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*PointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestIndirectCall2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::IndirectCallTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto deltaG1MemoryNode = pointsToGraph.getNodeForDelta(test.GetDeltaG1());
    auto deltaG2MemoryNode = pointsToGraph.getNodeForDelta(test.GetDeltaG2());

    auto allocaPxMemoryNode = pointsToGraph.getNodeForAlloca(test.GetAllocaPx());
    auto allocaPyMemoryNode = pointsToGraph.getNodeForAlloca(test.GetAllocaPy());
    auto allocaPzMemoryNode = pointsToGraph.getNodeForAlloca(test.GetAllocaPz());

    const jlm::util::HashSet pX = {
      allocaPxMemoryNode,
    };
    const jlm::util::HashSet pY = {
      allocaPyMemoryNode,
    };
    const jlm::util::HashSet pZ = {
      allocaPzMemoryNode,
    };
    const jlm::util::HashSet pXZ = { allocaPxMemoryNode, allocaPzMemoryNode };
    const jlm::util::HashSet pXYZG1G2 = { allocaPxMemoryNode,
                                          allocaPyMemoryNode,
                                          allocaPzMemoryNode,
                                          deltaG1MemoryNode,
                                          deltaG2MemoryNode };
    const jlm::util::HashSet pG1G2 = { deltaG1MemoryNode, deltaG2MemoryNode };

    /*
     * Validate function four()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaFour());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaFour());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function three()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaI());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetIndirectCall());
      EXPECT_TRUE(setsEqual(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaI());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function x()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaX());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, pXZ));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaX());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, pXZ));
    }

    /*
     * Validate function y()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaY());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, pY));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaY());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, pY));
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, pG1G2));

      auto & callXNodes = modRefSummary.GetSimpleNodeModRef(test.GetTestCallX());
      EXPECT_TRUE(setsEqual(callXNodes, pX));

      auto & callYNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallY());
      EXPECT_TRUE(setsEqual(callYNodes, pY));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, pG1G2));
    }

    /*
     * Validate function test2()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest2());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & callXNodes = modRefSummary.GetSimpleNodeModRef(test.GetTest2CallX());
      EXPECT_TRUE(setsEqual(callXNodes, pZ));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest2());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }
  };

  jlm::llvm::IndirectCallTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*PointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestGamma)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::GammaTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto externalMemoryNode = pointsToGraph.getExternalMemoryNode();

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaEntryNodes, { externalMemoryNode }));

    auto gammaEntryNodes = modRefSummary.GetGammaEntryModRef(*test.gamma);
    EXPECT_TRUE(setsEqual(gammaEntryNodes, {}));

    auto gammaExitNodes = modRefSummary.GetGammaExitModRef(*test.gamma);
    EXPECT_TRUE(setsEqual(gammaExitNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaExitNodes, { externalMemoryNode }));
  };

  jlm::llvm::GammaTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestTheta)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::ThetaTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto externalMemoryNode = pointsToGraph.getExternalMemoryNode();

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaEntryNodes, { externalMemoryNode }));

    auto & thetaEntryExitNodes = modRefSummary.GetThetaModRef(*test.theta);
    EXPECT_TRUE(setsEqual(thetaEntryExitNodes, { externalMemoryNode }));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    EXPECT_TRUE(setsEqual(lambdaExitNodes, { externalMemoryNode }));
  };

  jlm::llvm::ThetaTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestDelta1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::DeltaTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto deltaFNode = pointsToGraph.getNodeForDelta(*test.delta_f);

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { deltaFNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { deltaFNode }));
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { deltaFNode }));

      auto & callEntryNodes = modRefSummary.GetSimpleNodeModRef(test.CallG());
      EXPECT_TRUE(setsEqual(callEntryNodes, { deltaFNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { deltaFNode }));
    }
  };

  jlm::llvm::DeltaTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestDelta2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::DeltaTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto deltaD1Node = pointsToGraph.getNodeForDelta(*test.delta_d1);
    auto deltaD2Node = pointsToGraph.getNodeForDelta(*test.delta_d2);

    /*
     * Validate function f1
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f1);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { deltaD1Node }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { deltaD1Node }));
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { deltaD1Node, deltaD2Node }));

      auto & callEntryNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1());
      EXPECT_TRUE(setsEqual(callEntryNodes, { deltaD1Node }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { deltaD1Node, deltaD2Node }));
    }
  };

  jlm::llvm::DeltaTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestImports)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::ImportTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto importD1Node = pointsToGraph.getNodeForImport(*test.import_d1);
    auto importD2Node = pointsToGraph.getNodeForImport(*test.import_d2);

    /*
     * Validate function f1
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f1);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { importD1Node }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { importD1Node }));
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { importD1Node, importD2Node }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1());
      EXPECT_TRUE(setsEqual(callNodes, { importD1Node }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { importD1Node, importD2Node }));
    }
  };

  jlm::llvm::ImportTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestPhi1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::PhiTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto resultAllocaNode = pointsToGraph.getNodeForAlloca(*test.alloca);

    /*
     * Validate function fib
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_fib);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { resultAllocaNode }));

      auto & callFibM1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallFibm1());
      EXPECT_TRUE(setsEqual(callFibM1Nodes, { resultAllocaNode }));

      auto & callFibM2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallFibm2());
      EXPECT_TRUE(setsEqual(callFibM2Nodes, { resultAllocaNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_fib);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { resultAllocaNode }));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallFib());
      EXPECT_TRUE(setsEqual(callNodes, { resultAllocaNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }
  };

  jlm::llvm::PhiTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestPhi2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::PhiTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto pTestAllocaMemoryNode = pointsToGraph.getNodeForAlloca(test.GetPTestAlloca());
    auto paAllocaMemoryNode = pointsToGraph.getNodeForAlloca(test.GetPaAlloca());
    [[maybe_unused]] auto pbAllocaMemoryNode = pointsToGraph.getNodeForAlloca(test.GetPbAlloca());
    auto pcAllocaMemoryNode = pointsToGraph.getNodeForAlloca(test.GetPcAlloca());
    auto pdAllocaMemoryNode = pointsToGraph.getNodeForAlloca(test.GetPdAlloca());

    jlm::util::HashSet pTestAC({ pTestAllocaMemoryNode, paAllocaMemoryNode, pcAllocaMemoryNode });
    jlm::util::HashSet pTestBD({ pTestAllocaMemoryNode, pbAllocaMemoryNode, pdAllocaMemoryNode });
    jlm::util::HashSet pTestCD({ pTestAllocaMemoryNode, pcAllocaMemoryNode, pdAllocaMemoryNode });
    jlm::util::HashSet pTestAD({ pTestAllocaMemoryNode, paAllocaMemoryNode, pdAllocaMemoryNode });
    jlm::util::HashSet pTestACD(
        { pTestAllocaMemoryNode, paAllocaMemoryNode, pcAllocaMemoryNode, pdAllocaMemoryNode });

    /*
     * Validate function eight()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaEight());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaEight());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaI());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetIndirectCall());
      EXPECT_TRUE(setsEqual(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaI());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }

    /*
     * Validate function a()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaA());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, pTestCD));

      auto & callBNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallB());
      EXPECT_TRUE(setsEqual(callBNodes, pTestAD));

      auto & callDNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallD());
      EXPECT_TRUE(setsEqual(callDNodes, pTestAC));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaA());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, pTestCD));
    }

    /*
     * Validate function b()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaB());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, pTestAD));

      auto & callINodes = modRefSummary.GetSimpleNodeModRef(test.GetCallI());
      EXPECT_TRUE(setsEqual(callINodes, {}));

      auto & callCNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallC());
      EXPECT_TRUE(setsEqual(callCNodes, pTestBD));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaB());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, pTestAD));
    }

    /*
     * Validate function c()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaC());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, pTestBD));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromC());
      EXPECT_TRUE(setsEqual(callNodes, pTestCD));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaC());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, pTestBD));
    }

    /*
     * Validate function d()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaD());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, pTestAC));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromD());
      EXPECT_TRUE(setsEqual(callNodes, pTestCD));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaD());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, pTestAC));
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromTest());
      EXPECT_TRUE(setsEqual(callNodes, { pTestAllocaMemoryNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, {}));
    }
  };

  jlm::llvm::PhiTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestPhiWithDelta)
{
  // Assert
  jlm::llvm::PhiWithDeltaTest test;
  std::unordered_map<const jlm::rvsdg::Output *, std::string> outputMap;
  // std::cout << jlm::rvsdg::view(&test.graph().GetRootRegion(), outputMap) << std::flush;

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph, outputMap) << std::flush;

  // Act
  auto modRefSummary =
      jlm::llvm::aa::RegionAwareModRefSummarizer::Create(test.module(), *pointsToGraph);

  // Assert
  // Nothing needs to be validated as there are only phi and delta nodes in the RVSDG.
}

TEST(RegionAwareModRefSummarizerTests, TestMemcpy)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::MemcpyTest & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto localArrayMemoryNode = pointsToGraph.getNodeForDelta(test.LocalArray());
    auto globalArrayMemoryNode = pointsToGraph.getNodeForDelta(test.GlobalArray());

    /*
     * Validate function f
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaF());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { globalArrayMemoryNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaF());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { globalArrayMemoryNode }));
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaG());
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { localArrayMemoryNode, globalArrayMemoryNode }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallF());
      EXPECT_TRUE(setsEqual(callNodes, { globalArrayMemoryNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaG());
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { localArrayMemoryNode, globalArrayMemoryNode }));
    }
  };

  jlm::llvm::MemcpyTest test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*PointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestEscapedMemory1)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::EscapedMemoryTest1 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto deltaAMemoryNode = pointsToGraph.getNodeForDelta(*test.DeltaA);
    auto deltaBMemoryNode = pointsToGraph.getNodeForDelta(*test.DeltaB);
    auto deltaXMemoryNode = pointsToGraph.getNodeForDelta(*test.DeltaX);
    auto deltaYMemoryNode = pointsToGraph.getNodeForDelta(*test.DeltaY);
    auto externalMemoryNode = pointsToGraph.getExternalMemoryNode();

    jlm::util::HashSet expectedMemoryNodes{ deltaAMemoryNode,
                                            deltaBMemoryNode,
                                            deltaXMemoryNode,
                                            deltaYMemoryNode,
                                            externalMemoryNode };

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.LambdaTest);
    EXPECT_TRUE(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.LambdaTest);
    EXPECT_TRUE(setsEqual(lambdaExitNodes, expectedMemoryNodes));
  };

  jlm::llvm::EscapedMemoryTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestEscapedMemory2)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::EscapedMemoryTest2 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto returnAddressMallocMemoryNode = pointsToGraph.getNodeForMalloc(*test.ReturnAddressMalloc);
    auto callExternalFunction1MallocMemoryNode =
        pointsToGraph.getNodeForMalloc(*test.CallExternalFunction1Malloc);

    auto externalMemoryNode = pointsToGraph.getExternalMemoryNode();

    /*
     * Validate ReturnAddress function
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.ReturnAddressFunction);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, { returnAddressMallocMemoryNode }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.ReturnAddressFunction);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, { returnAddressMallocMemoryNode }));
    }

    /*
     * Validate CallExternalFunction1 function
     */
    {
      jlm::util::HashSet expectedMemoryNodes{ returnAddressMallocMemoryNode,
                                              callExternalFunction1MallocMemoryNode,
                                              externalMemoryNode };

      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.CallExternalFunction1);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.ExternalFunction1Call);
      EXPECT_TRUE(setsEqual(callNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.CallExternalFunction1);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }

    /*
     * Validate CallExternalFunction2 function
     */
    {
      jlm::util::HashSet<jlm::llvm::aa::PointsToGraph::NodeIndex> expectedMemoryNodes{
        returnAddressMallocMemoryNode,
        callExternalFunction1MallocMemoryNode,
        externalMemoryNode
      };

      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.CallExternalFunction2);
      EXPECT_TRUE(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.ExternalFunction2Call);
      EXPECT_TRUE(setsEqual(callNodes, expectedMemoryNodes));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.CallExternalFunction2);
      EXPECT_TRUE(setsEqual(lambdaExitNodes, expectedMemoryNodes));
    }
  };

  jlm::llvm::EscapedMemoryTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, TestEscapedMemory3)
{
  /*
   * Arrange
   */
  auto ValidateProvider = [](const jlm::llvm::EscapedMemoryTest3 & test,
                             const jlm::llvm::aa::ModRefSummary & modRefSummary,
                             const jlm::llvm::aa::PointsToGraph & pointsToGraph)
  {
    auto deltaMemoryNode = pointsToGraph.getNodeForDelta(*test.DeltaGlobal);
    auto externalMemoryNode = pointsToGraph.getExternalMemoryNode();

    jlm::util::HashSet expectedMemoryNodes{ deltaMemoryNode, externalMemoryNode };

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.LambdaTest);
    EXPECT_TRUE(setsEqual(lambdaEntryNodes, expectedMemoryNodes));

    auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.CallExternalFunction);
    EXPECT_TRUE(setsEqual(callNodes, expectedMemoryNodes));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.LambdaTest);
    EXPECT_TRUE(setsEqual(lambdaExitNodes, expectedMemoryNodes));
  };

  jlm::llvm::EscapedMemoryTest3 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunAndersen(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

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

TEST(RegionAwareModRefSummarizerTests, testSetjmpHandling)
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Creates the RVSDG equivalent of the program
  //
  // void opaque();
  // int _setjmp(jmp_buf*);
  //
  // jmp_buf buf;
  //
  // static void h() {
  //     opaque(); // This call should have a in its Mod/Ref set
  // }
  //
  // static void k() {
  //     // This call does nothing
  // }
  //
  // static void g(int* p) {
  //     if (_setjmp(&buf))
  //         return;
  //     else {
  //         *p = 10;
  //         h(); // This call should have a in its Mod/Ref set
  //         k(); // Nothing should be routed into this call
  //     }
  // }
  //
  // int f() {
  //     int a;
  //     g(a);
  //     return a;
  // }

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto & rootRegion = graph.GetRootRegion();

  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto pointerType = PointerType::Create();
  const auto int32Type = rvsdg::BitType::Create(32);
  // We don't care about the type of the jmp_buf, just use an array
  const auto jmpBufType = ArrayType::Create(int32Type, 34);
  const auto unitType = rvsdg::UnitType::Create();

  const auto unitFunctionType = rvsdg::FunctionType::Create(
      { ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  const auto setjmpFunctionType = rvsdg::FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { int32Type, ioStateType, memoryStateType });

  const auto gFunctionType = rvsdg::FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  const auto fFunctionType = rvsdg::FunctionType::Create(
      { ioStateType, memoryStateType },
      { int32Type, ioStateType, memoryStateType });

  auto & opaqueImport = LlvmGraphImport::Create(
      graph,
      unitFunctionType,
      unitFunctionType,
      "opaque",
      Linkage::externalLinkage);

  auto & setjmpImport = LlvmGraphImport::Create(
      graph,
      setjmpFunctionType,
      setjmpFunctionType,
      "_setjmp",
      Linkage::externalLinkage);

  auto & bufGlobal = *rvsdg::DeltaNode::Create(
      &rootRegion,
      DeltaOperation::Create(jmpBufType, "buf", Linkage::externalLinkage, "", false));
  bufGlobal.finalize(UndefValueOperation::Create(*bufGlobal.subregion(), jmpBufType));

  rvsdg::SimpleNode * callOpaqueNode = nullptr;
  rvsdg::SimpleNode * callHNode = nullptr;
  rvsdg::SimpleNode * callKNode = nullptr;
  rvsdg::SimpleNode * allocaNode = nullptr;

  auto & hLambdaNode = *rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(unitFunctionType, "h", Linkage::internalLinkage));
  {
    const auto arguments = hLambdaNode.GetFunctionArguments();
    auto ioState = arguments.at(0);
    auto memoryState = arguments.at(1);

    const auto opaqueCtxVar = hLambdaNode.AddContextVar(opaqueImport);

    const auto call =
        CallOperation::Create(opaqueCtxVar.inner, unitFunctionType, { ioState, memoryState });
    callOpaqueNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*call[0]);
    ioState = call[0];
    memoryState = call[1];

    hLambdaNode.finalize({ ioState, memoryState });
  }

  auto & kLambdaNode = *rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(unitFunctionType, "k", Linkage::internalLinkage));
  {
    const auto arguments = kLambdaNode.GetFunctionArguments();
    kLambdaNode.finalize({ arguments.at(0), arguments.at(1) });
  }

  auto & gLambdaNode = *rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(gFunctionType, "g", Linkage::internalLinkage));
  {
    const auto arguments = gLambdaNode.GetFunctionArguments();
    const auto p = arguments.at(0);
    auto ioState = arguments.at(1);
    auto memoryState = arguments.at(2);

    const auto setjmpCtxVar = gLambdaNode.AddContextVar(setjmpImport);
    const auto bufCtxVar = gLambdaNode.AddContextVar(bufGlobal.output());
    const auto hCtxVar = gLambdaNode.AddContextVar(*hLambdaNode.output());
    const auto kCtxVar = gLambdaNode.AddContextVar(*kLambdaNode.output());

    const auto setjmpCall = CallOperation::Create(
        setjmpCtxVar.inner,
        setjmpFunctionType,
        { bufCtxVar.inner, ioState, memoryState });
    auto & setjmpResult = *setjmpCall[0];
    ioState = setjmpCall[1];
    memoryState = setjmpCall[2];

    auto & matchOutput = *rvsdg::MatchOperation::Create(setjmpResult, { { 0, 0 } }, 1, 2);
    auto & gammaNode = rvsdg::GammaNode::Create(matchOutput, 2, { unitType, unitType });
    auto pEntryVar = gammaNode.AddEntryVar(p);
    auto hEntryVar = gammaNode.AddEntryVar(hCtxVar.inner);
    auto kEntryVar = gammaNode.AddEntryVar(kCtxVar.inner);
    auto ioStateEntryVar = gammaNode.AddEntryVar(ioState);
    auto memoryStateEntryVar = gammaNode.AddEntryVar(memoryState);
    auto & elseRegion = *gammaNode.subregion(0);
    const auto constant10 = IntegerConstantOperation::Create(elseRegion, 32, 10).output(0);
    const auto storeOutputs = StoreNonVolatileOperation::Create(
        pEntryVar.branchArgument[0],
        constant10,
        { memoryStateEntryVar.branchArgument[0] },
        4);

    const auto hCall = CallOperation::Create(
        hEntryVar.branchArgument[0],
        unitFunctionType,
        { ioStateEntryVar.branchArgument[0], storeOutputs[0] });
    callHNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*hCall[0]);

    const auto kCall = CallOperation::Create(
        kEntryVar.branchArgument[0],
        unitFunctionType,
        { hCall[0], hCall[1] });
    callKNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*kCall[0]);

    ioState = gammaNode.AddExitVar({ kCall[0], ioStateEntryVar.branchArgument[1] }).output;
    memoryState = gammaNode.AddExitVar({ kCall[1], memoryStateEntryVar.branchArgument[1] }).output;

    gLambdaNode.finalize({ ioState, memoryState });
  }

  auto & fLambdaNode = *rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(fFunctionType, "f", Linkage::externalLinkage));
  {
    const auto arguments = fLambdaNode.GetFunctionArguments();
    const auto ioStateIn = arguments.at(0);
    const auto memoryStateIn = arguments.at(1);

    const auto gCtxVar = fLambdaNode.AddContextVar(*gLambdaNode.output());

    const auto constant1 =
        IntegerConstantOperation::Create(*fLambdaNode.subregion(), 32, 1).output(0);
    const auto aAlloca = AllocaOperation::create(int32Type, constant1, 4);
    allocaNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*aAlloca[0]);

    auto & memoryStateJoin =
        rvsdg::CreateOpNode<MemoryStateJoinOperation>({ memoryStateIn, aAlloca[1] }, 2);

    const auto gCall = CallOperation::Create(
        gCtxVar.inner,
        gFunctionType,
        { aAlloca[0], ioStateIn, memoryStateJoin.output(0) });

    auto loadOutputs = LoadNonVolatileOperation::Create(aAlloca[0], { gCall[1] }, int32Type, 4);

    fLambdaNode.finalize({ loadOutputs[0], gCall[0], loadOutputs[1] });
  }

  rvsdg::GraphExport::Create(*fLambdaNode.output(), "f");

  util::graph::Writer gw;
  LlvmDotWriter writer;
  writer.WriteGraphs(gw, rootRegion, true);
  // gw.outputAllGraphs(std::cout, util::graph::OutputFormat::Dot);

  // Act
  util::StatisticsCollectorSettings settings({ util::Statistics::Id::RegionAwareModRefSummarizer });
  util::StatisticsCollector collector(settings);
  const auto ptg = RunAndersen(rvsdgModule);
  const auto modRefSummary = aa::RegionAwareModRefSummarizer::Create(rvsdgModule, *ptg, collector);

  // Assert
  EXPECT_NE(callOpaqueNode, nullptr);
  EXPECT_NE(callHNode, nullptr);
  EXPECT_NE(callKNode, nullptr);
  EXPECT_NE(allocaNode, nullptr);

  const auto allocaPtgNode = ptg->getNodeForAlloca(*allocaNode);

  // The call to h() within g() should contain a in its Mod/Ref set
  const auto callHModRef = modRefSummary->GetSimpleNodeModRef(*callHNode);
  EXPECT_TRUE(callHModRef.Contains(allocaPtgNode));

  // The call to k() should NOT contain a in its Mod/Ref set
  const auto callKModRef = modRefSummary->GetSimpleNodeModRef(*callKNode);
  EXPECT_FALSE(callKModRef.Contains(allocaPtgNode));

  // The call to opaque() within h() should contain a in its Mod/Ref set
  const auto callOpaqueModRef = modRefSummary->GetSimpleNodeModRef(*callOpaqueNode);
  EXPECT_TRUE(callOpaqueModRef.Contains(allocaPtgNode));

  // Check the statistics to ensure that the right functions in the call graph were marked
  auto & statistic = *collector.CollectedStatistics().begin();
  // Only k() is not in the same SCC as <external>
  EXPECT_EQ(statistic.GetMeasurementValue<uint64_t>("#CallGraphSccs"), 2u);
  // g(), k() and h() are the only functions within an active setjmp
  EXPECT_EQ(statistic.GetMeasurementValue<uint64_t>("#FunctionsCallingSetjmp"), 1u);
}

TEST(RegionAwareModRefSummarizerTests, TestStatistics)
{
  using namespace jlm;

  // Arrange
  jlm::llvm::LoadTest2 test;
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
  EXPECT_EQ(statisticsCollector.NumCollectedStatistics(), 1u);
  auto & statistics = *statisticsCollector.CollectedStatistics().begin();

  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#RvsdgNodes"), 18u);
  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#RvsdgRegions"), 2u);
  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#PointsToGraphMemoryNodes"), 7u);
  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#SimpleAllocas"), 5u);
  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#NonReentrantAllocas"), 5u);
  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#CallGraphSccs"), 2u);

  EXPECT_TRUE(statistics.HasTimer("CallGraphTimer"));
  EXPECT_TRUE(statistics.HasTimer("AllocasDeadInSccsTimer"));
  EXPECT_TRUE(statistics.HasTimer("SimpleAllocasSetTimer"));
  EXPECT_TRUE(statistics.HasTimer("NonReentrantAllocaSetsTimer"));
  EXPECT_TRUE(statistics.HasTimer("CreateExternalModRefSetTimer"));
  EXPECT_TRUE(statistics.HasTimer("AnnotationTimer"));
  EXPECT_TRUE(statistics.HasTimer("SolvingTimer"));
}
