/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/DotWriter.hpp>
#include <jlm/llvm/ir/CallingConvention.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/UnitType.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Statistics.hpp>
#include <unordered_map>

using jlm::llvm::aa::ModRefEffect;
using NodeIndex = jlm::llvm::aa::PointsToGraph::NodeIndex;

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunAndersen(jlm::llvm::LlvmRvsdgModule & rvsdgModule)
{
  jlm::llvm::aa::Andersen andersen;
  return andersen.Analyze(rvsdgModule);
}

static const char *
effectToString(ModRefEffect effect)
{
  switch (effect)
  {
  case ModRefEffect::ModRef:
    return "ModRef";
  case ModRefEffect::ModOnly:
    return "ModOnly";
  case ModRefEffect::RefOnly:
    return "RefOnly";
  case ModRefEffect::NoEffect:
    return "NoEffect";
  default:
    JLM_UNREACHABLE("Unknown effect");
  }
}

// Helper for creating a map where all memory nodes are mapped to the same effect
static std::unordered_map<NodeIndex, ModRefEffect>
allWithEffect(const jlm::util::HashSet<NodeIndex> & memoryNodes, ModRefEffect effect)
{
  std::unordered_map<NodeIndex, ModRefEffect> expectedEffects;
  for (auto memoryNode : memoryNodes.Items())
    expectedEffects.emplace(memoryNode, effect);
  return expectedEffects;
}

// Helper for comparing HashSets of MemoryNodes without needing explicit constructors
static bool
assertSetContains(
    const jlm::llvm::aa::ModRefSet & receivedMemoryNodes,
    std::unordered_map<NodeIndex, ModRefEffect> expectedEffects)
{
  bool result = true;

  for (auto [memoryNode, modRefEffect] : receivedMemoryNodes.getModRefNodes())
  {
    auto it = expectedEffects.find(memoryNode);
    if (it == expectedEffects.end())
    {
      result = false;
      std::cerr << "ModRefSet contained unexpected node " << memoryNode << " with effect "
                << effectToString(modRefEffect) << std::endl;
      continue;
    }

    if (it->second != modRefEffect)
    {
      result = false;
      std::cerr << "ModRefSet contained " << memoryNode << " with effect "
                << effectToString(modRefEffect) << " but expected effect "
                << effectToString(it->second) << std::endl;
    }
    expectedEffects.erase(it);
  }

  // Any remaining expected memory nodes indicate an error
  for (auto [memoryNode, modRefEffect] : expectedEffects)
  {
    result = false;
    std::cerr << "ModRefSet did not contain node " << memoryNode << " with expected effect "
              << effectToString(modRefEffect) << std::endl;
  }

  return result;
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

    // Every alloca in the lambda is non-reentrant, so they are left out of the lambda ModRefSet
    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));

    auto storeANode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(test.alloca_a->output(0)->SingleUser());
    EXPECT_TRUE(jlm::rvsdg::is<jlm::llvm::StoreNonVolatileOperation>(storeANode));

    auto & storeANodes = modRefSummary.GetSimpleNodeModRef(*storeANode);
    ASSERT_TRUE(assertSetContains(storeANodes, { { allocaAMemoryNode, ModRefEffect::ModOnly } }));
  };

  jlm::llvm::StoreTest1 test;
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
    ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
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

    // Since the function only contains loads, the external memory node is RefOnly
    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda);
    ASSERT_TRUE(
        assertSetContains(lambdaEntryNodes, { { externalMemoryNode, ModRefEffect::RefOnly } }));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    ASSERT_TRUE(
        assertSetContains(lambdaExitNodes, { { externalMemoryNode, ModRefEffect::RefOnly } }));

    // The loads from *p and **p (aka *x) should both only reference external memory nodes
    auto & loadPModRefSet = modRefSummary.GetSimpleNodeModRef(
        *jlm::util::assertedCast<jlm::rvsdg::SimpleNode>(test.load_p));
    ASSERT_TRUE(
        assertSetContains(loadPModRefSet, { { externalMemoryNode, ModRefEffect::RefOnly } }));

    auto & loadXModRefSet = modRefSummary.GetSimpleNodeModRef(
        *jlm::util::assertedCast<jlm::rvsdg::SimpleNode>(test.load_x));
    ASSERT_TRUE(
        assertSetContains(loadXModRefSet, { { externalMemoryNode, ModRefEffect::RefOnly } }));
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
    ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
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
    auto & numLambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.Lambda());
    auto & numLambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.Lambda());

    EXPECT_EQ(numLambdaEntryNodes.getModRefNodes().size(), 0u);
    EXPECT_EQ(numLambdaExitNodes.getModRefNodes().size(), 0u);
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
      ASSERT_TRUE(assertSetContains(
          lambdaFEntryNodes,
          { { allocaXMemoryNode, ModRefEffect::RefOnly },
            { allocaYMemoryNode, ModRefEffect::RefOnly } }));

      auto & lambdaFExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f);
      ASSERT_TRUE(assertSetContains(
          lambdaFExitNodes,
          { { allocaXMemoryNode, ModRefEffect::RefOnly },
            { allocaYMemoryNode, ModRefEffect::RefOnly } }));
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaGEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g);
      ASSERT_TRUE(
          assertSetContains(lambdaGEntryNodes, { { allocaZMemoryNode, ModRefEffect::RefOnly } }));

      auto & lambdaGExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g);
      ASSERT_TRUE(
          assertSetContains(lambdaGExitNodes, { { allocaZMemoryNode, ModRefEffect::RefOnly } }));
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaHEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h);
      ASSERT_TRUE(assertSetContains(lambdaHEntryNodes, {}));

      auto & callFNodes = modRefSummary.GetSimpleNodeModRef(test.CallF());
      ASSERT_TRUE(assertSetContains(
          callFNodes,
          { { allocaXMemoryNode, ModRefEffect::RefOnly },
            { allocaYMemoryNode, ModRefEffect::RefOnly } }));

      auto & callGNodes = modRefSummary.GetSimpleNodeModRef(test.CallG());
      ASSERT_TRUE(assertSetContains(callGNodes, { { allocaZMemoryNode, ModRefEffect::RefOnly } }));

      auto & lambdaHExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h);
      ASSERT_TRUE(assertSetContains(lambdaHExitNodes, {}));
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
      ASSERT_TRUE(assertSetContains(
          lambdaCreateEntryNodes,
          { { mallocMemoryNode, ModRefEffect::RefOnly } }));

      auto & lambdaCreateExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_create);
      ASSERT_TRUE(assertSetContains(
          lambdaCreateExitNodes,
          { { mallocMemoryNode, ModRefEffect::RefOnly } }));
    }

    /*
     * Validate function destroy
     */
    {
      auto & lambdaDestroyEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_destroy);
      ASSERT_TRUE(assertSetContains(
          lambdaDestroyEntryNodes,
          { { mallocMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaDestroyExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_destroy);
      ASSERT_TRUE(assertSetContains(
          lambdaDestroyExitNodes,
          { { mallocMemoryNode, ModRefEffect::ModOnly } }));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaTestEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test);
      ASSERT_TRUE(
          assertSetContains(lambdaTestEntryNodes, { { mallocMemoryNode, ModRefEffect::ModRef } }));

      auto & callCreate1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate1());
      ASSERT_TRUE(
          assertSetContains(callCreate1Nodes, { { mallocMemoryNode, ModRefEffect::RefOnly } }));

      auto & callCreate2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallCreate2());
      ASSERT_TRUE(
          assertSetContains(callCreate2Nodes, { { mallocMemoryNode, ModRefEffect::RefOnly } }));

      auto & callDestroy1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy1());
      ASSERT_TRUE(
          assertSetContains(callDestroy1Nodes, { { mallocMemoryNode, ModRefEffect::ModOnly } }));

      auto & callDestroy2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallDestroy2());
      ASSERT_TRUE(
          assertSetContains(callDestroy2Nodes, { { mallocMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaTestExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test);
      ASSERT_TRUE(
          assertSetContains(lambdaTestExitNodes, { { mallocMemoryNode, ModRefEffect::ModRef } }));
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
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaFour());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function three
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function indcall
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaIndcall());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallIndcall());
      ASSERT_TRUE(assertSetContains(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaIndcall());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & callFourNodes = modRefSummary.GetSimpleNodeModRef(test.CallFour());
      ASSERT_TRUE(assertSetContains(callFourNodes, {}));

      auto & callThreeNodes = modRefSummary.GetSimpleNodeModRef(test.CallThree());
      ASSERT_TRUE(assertSetContains(callThreeNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
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
    auto allocaPxMemoryNode = pointsToGraph.getNodeForAlloca(test.GetAllocaPx());
    auto allocaPyMemoryNode = pointsToGraph.getNodeForAlloca(test.GetAllocaPy());
    auto allocaPzMemoryNode = pointsToGraph.getNodeForAlloca(test.GetAllocaPz());

    /*
     * Validate function four()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaFour());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaFour());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function three()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaThree());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaThree());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaI());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetIndirectCall());
      ASSERT_TRUE(assertSetContains(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaI());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function x()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaX());
      ASSERT_TRUE(assertSetContains(
          lambdaEntryNodes,
          { { allocaPxMemoryNode, ModRefEffect::ModOnly },
            { allocaPzMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaX());
      ASSERT_TRUE(assertSetContains(
          lambdaExitNodes,
          { { allocaPxMemoryNode, ModRefEffect::ModOnly },
            { allocaPzMemoryNode, ModRefEffect::ModOnly } }));
    }

    /*
     * Validate function y()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaY());
      ASSERT_TRUE(
          assertSetContains(lambdaEntryNodes, { { allocaPyMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaY());
      ASSERT_TRUE(
          assertSetContains(lambdaExitNodes, { { allocaPyMemoryNode, ModRefEffect::ModOnly } }));
    }

    /*
     * Validate function test()
     */
    {
      // g1 and g2 are effectively read-only so they get omitted from the lambda's set
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & callXNodes = modRefSummary.GetSimpleNodeModRef(test.GetTestCallX());
      ASSERT_TRUE(assertSetContains(callXNodes, { { allocaPxMemoryNode, ModRefEffect::ModOnly } }));

      auto & callYNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallY());
      ASSERT_TRUE(assertSetContains(callYNodes, { { allocaPyMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function test2()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest2());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & callXNodes = modRefSummary.GetSimpleNodeModRef(test.GetTest2CallX());
      ASSERT_TRUE(assertSetContains(callXNodes, { { allocaPzMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest2());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
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
    ASSERT_TRUE(
        assertSetContains(lambdaEntryNodes, { { externalMemoryNode, ModRefEffect::RefOnly } }));

    auto & gammaEntryNodes = modRefSummary.GetGammaEntryModRef(*test.gamma);
    ASSERT_TRUE(assertSetContains(gammaEntryNodes, {}));

    auto & gammaExitNodes = modRefSummary.GetGammaExitModRef(*test.gamma);
    ASSERT_TRUE(assertSetContains(gammaExitNodes, {}));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    ASSERT_TRUE(
        assertSetContains(lambdaExitNodes, { { externalMemoryNode, ModRefEffect::RefOnly } }));
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
    ASSERT_TRUE(
        assertSetContains(lambdaEntryNodes, { { externalMemoryNode, ModRefEffect::ModOnly } }));

    auto & thetaEntryExitNodes = modRefSummary.GetThetaModRef(*test.theta);
    ASSERT_TRUE(
        assertSetContains(thetaEntryExitNodes, { { externalMemoryNode, ModRefEffect::ModOnly } }));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda);
    ASSERT_TRUE(
        assertSetContains(lambdaExitNodes, { { externalMemoryNode, ModRefEffect::ModOnly } }));
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
      // g() only reads, and it does not get any unknown pointers
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_g);
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, { { deltaFNode, ModRefEffect::RefOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_g);
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, { { deltaFNode, ModRefEffect::RefOnly } }));
    }

    /*
     * Validate function h
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_h);
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, { { deltaFNode, ModRefEffect::ModRef } }));

      auto & callEntryNodes = modRefSummary.GetSimpleNodeModRef(test.CallG());
      ASSERT_TRUE(assertSetContains(callEntryNodes, { { deltaFNode, ModRefEffect::RefOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_h);
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, { { deltaFNode, ModRefEffect::ModRef } }));
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
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, { { deltaD1Node, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1);
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, { { deltaD1Node, ModRefEffect::ModOnly } }));
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2);
      ASSERT_TRUE(assertSetContains(
          lambdaEntryNodes,
          { { deltaD1Node, ModRefEffect::ModOnly }, { deltaD2Node, ModRefEffect::ModOnly } }));

      auto & callEntryNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1());
      ASSERT_TRUE(assertSetContains(callEntryNodes, { { deltaD1Node, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2);
      ASSERT_TRUE(assertSetContains(
          lambdaExitNodes,
          { { deltaD1Node, ModRefEffect::ModOnly }, { deltaD2Node, ModRefEffect::ModOnly } }));
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
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, { { importD1Node, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f1);
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, { { importD1Node, ModRefEffect::ModOnly } }));
    }

    /*
     * Validate function f2
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_f2);
      ASSERT_TRUE(assertSetContains(
          lambdaEntryNodes,
          { { importD1Node, ModRefEffect::ModOnly }, { importD2Node, ModRefEffect::ModOnly } }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallF1());
      ASSERT_TRUE(assertSetContains(callNodes, { { importD1Node, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_f2);
      ASSERT_TRUE(assertSetContains(
          lambdaExitNodes,
          { { importD1Node, ModRefEffect::ModOnly }, { importD2Node, ModRefEffect::ModOnly } }));
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
      ASSERT_TRUE(
          assertSetContains(lambdaEntryNodes, { { resultAllocaNode, ModRefEffect::ModRef } }));

      auto & callFibM1Nodes = modRefSummary.GetSimpleNodeModRef(test.CallFibm1());
      ASSERT_TRUE(
          assertSetContains(callFibM1Nodes, { { resultAllocaNode, ModRefEffect::ModRef } }));

      auto & callFibM2Nodes = modRefSummary.GetSimpleNodeModRef(test.CallFibm2());
      ASSERT_TRUE(
          assertSetContains(callFibM2Nodes, { { resultAllocaNode, ModRefEffect::ModRef } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_fib);
      ASSERT_TRUE(
          assertSetContains(lambdaExitNodes, { { resultAllocaNode, ModRefEffect::ModRef } }));
    }

    /*
     * Validate function test
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.lambda_test);
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallFib());
      ASSERT_TRUE(assertSetContains(callNodes, { { resultAllocaNode, ModRefEffect::ModRef } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.lambda_test);
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
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
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaEight());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function i()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaI());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetIndirectCall());
      ASSERT_TRUE(assertSetContains(callNodes, {}));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaI());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
    }

    /*
     * Validate function a()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaA());
      ASSERT_TRUE(
          assertSetContains(lambdaEntryNodes, allWithEffect(pTestCD, ModRefEffect::ModOnly)));

      auto & callBNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallB());
      ASSERT_TRUE(assertSetContains(callBNodes, { { paAllocaMemoryNode, ModRefEffect::ModOnly } }));

      auto & callDNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallD());
      ASSERT_TRUE(assertSetContains(callDNodes, { { paAllocaMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaA());
      ASSERT_TRUE(
          assertSetContains(lambdaExitNodes, allWithEffect(pTestCD, ModRefEffect::ModOnly)));
    }

    /*
     * Validate function b()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaB());
      ASSERT_TRUE(
          assertSetContains(lambdaEntryNodes, { { paAllocaMemoryNode, ModRefEffect::ModOnly } }));

      auto & callINodes = modRefSummary.GetSimpleNodeModRef(test.GetCallI());
      ASSERT_TRUE(assertSetContains(callINodes, {}));

      auto & callCNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallC());
      ASSERT_TRUE(assertSetContains(callCNodes, { { pbAllocaMemoryNode, ModRefEffect::ModRef } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaB());
      ASSERT_TRUE(
          assertSetContains(lambdaExitNodes, { { paAllocaMemoryNode, ModRefEffect::ModOnly } }));
    }

    /*
     * Validate function c()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaC());
      ASSERT_TRUE(
          assertSetContains(lambdaEntryNodes, { { pbAllocaMemoryNode, ModRefEffect::ModRef } }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromC());
      ASSERT_TRUE(assertSetContains(callNodes, { { pcAllocaMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaC());
      ASSERT_TRUE(
          assertSetContains(lambdaExitNodes, { { pbAllocaMemoryNode, ModRefEffect::ModRef } }));
    }

    /*
     * Validate function d()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaD());
      ASSERT_TRUE(
          assertSetContains(lambdaEntryNodes, { { paAllocaMemoryNode, ModRefEffect::ModOnly } }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromD());
      ASSERT_TRUE(assertSetContains(callNodes, { { pdAllocaMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaD());
      ASSERT_TRUE(
          assertSetContains(lambdaExitNodes, { { paAllocaMemoryNode, ModRefEffect::ModOnly } }));
    }

    /*
     * Validate function test()
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.GetLambdaTest());
      ASSERT_TRUE(assertSetContains(lambdaEntryNodes, {}));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.GetCallAFromTest());
      ASSERT_TRUE(
          assertSetContains(callNodes, { { pTestAllocaMemoryNode, ModRefEffect::ModOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.GetLambdaTest());
      ASSERT_TRUE(assertSetContains(lambdaExitNodes, {}));
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
    auto initArrayMemoryNode = pointsToGraph.getNodeForDelta(test.InitArray());
    auto globalArrayMemoryNode = pointsToGraph.getNodeForDelta(test.GlobalArray());

    /*
     * Validate function f
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaF());
      ASSERT_TRUE(
          assertSetContains(lambdaEntryNodes, { { globalArrayMemoryNode, ModRefEffect::ModRef } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaF());
      ASSERT_TRUE(
          assertSetContains(lambdaExitNodes, { { globalArrayMemoryNode, ModRefEffect::ModRef } }));
    }

    /*
     * Validate function g
     */
    {
      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(test.LambdaG());
      EXPECT_TRUE(assertSetContains(
          lambdaEntryNodes,
          { { globalArrayMemoryNode, ModRefEffect::ModRef },
            { initArrayMemoryNode, ModRefEffect::RefOnly } }));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(test.CallF());
      ASSERT_TRUE(
          assertSetContains(callNodes, { { globalArrayMemoryNode, ModRefEffect::ModRef } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(test.LambdaG());
      EXPECT_TRUE(assertSetContains(
          lambdaExitNodes,
          { { globalArrayMemoryNode, ModRefEffect::ModRef },
            { initArrayMemoryNode, ModRefEffect::RefOnly } }));
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
    auto deltaBMemoryNode = pointsToGraph.getNodeForDelta(*test.DeltaB);
    // Delta A, X and Y have been compressed into the external memory node
    auto externalMemoryNode = pointsToGraph.getExternalMemoryNode();

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.LambdaTest);
    ASSERT_TRUE(assertSetContains(
        lambdaEntryNodes,
        { { deltaBMemoryNode, ModRefEffect::ModOnly },
          { externalMemoryNode, ModRefEffect::RefOnly } }));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.LambdaTest);
    ASSERT_TRUE(assertSetContains(
        lambdaExitNodes,
        { { deltaBMemoryNode, ModRefEffect::ModOnly },
          { externalMemoryNode, ModRefEffect::RefOnly } }));
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
      ASSERT_TRUE(assertSetContains(
          lambdaEntryNodes,
          { { returnAddressMallocMemoryNode, ModRefEffect::RefOnly } }));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.ReturnAddressFunction);
      ASSERT_TRUE(assertSetContains(
          lambdaExitNodes,
          { { returnAddressMallocMemoryNode, ModRefEffect::RefOnly } }));
    }

    /*
     * Validate CallExternalFunction1 function
     */
    {
      // The returnAddressMallocMemoryNode is compressed into the external node
      jlm::util::HashSet expectedMemoryNodes{ callExternalFunction1MallocMemoryNode,
                                              externalMemoryNode };

      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.CallExternalFunction1);
      ASSERT_TRUE(assertSetContains(
          lambdaEntryNodes,
          allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.ExternalFunction1Call);
      ASSERT_TRUE(
          assertSetContains(callNodes, allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.CallExternalFunction1);
      ASSERT_TRUE(assertSetContains(
          lambdaExitNodes,
          allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));
    }

    /*
     * Validate CallExternalFunction2 function
     */
    {
      // The function only does a call, and a load of unknown, so everything can be compressed
      jlm::util::HashSet<jlm::llvm::aa::PointsToGraph::NodeIndex> expectedMemoryNodes{
        externalMemoryNode
      };

      auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.CallExternalFunction2);
      ASSERT_TRUE(assertSetContains(
          lambdaEntryNodes,
          allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));

      auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.ExternalFunction2Call);
      ASSERT_TRUE(
          assertSetContains(callNodes, allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));

      auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.CallExternalFunction2);
      ASSERT_TRUE(assertSetContains(
          lambdaExitNodes,
          allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));
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
    auto externalMemoryNode = pointsToGraph.getExternalMemoryNode();

    // DeltaGlobal has been compressed into the externalMemoryNode
    jlm::util::HashSet expectedMemoryNodes{ externalMemoryNode };

    auto & lambdaEntryNodes = modRefSummary.GetLambdaEntryModRef(*test.LambdaTest);
    ASSERT_TRUE(assertSetContains(
        lambdaEntryNodes,
        allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));

    auto & callNodes = modRefSummary.GetSimpleNodeModRef(*test.CallExternalFunction);
    ASSERT_TRUE(
        assertSetContains(callNodes, allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));

    auto & lambdaExitNodes = modRefSummary.GetLambdaExitModRef(*test.LambdaTest);
    ASSERT_TRUE(assertSetContains(
        lambdaExitNodes,
        allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));
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

  auto & opaqueImport = LlvmGraphImport::createFunctionImport(
      graph,
      unitFunctionType,
      "opaque",
      Linkage::externalLinkage,
      CallingConvention::Default);

  auto & setjmpImport = LlvmGraphImport::createFunctionImport(
      graph,
      setjmpFunctionType,
      "_setjmp",
      Linkage::externalLinkage,
      CallingConvention::Default);

  auto & bufGlobal = *rvsdg::DeltaNode::Create(
      &rootRegion,
      LlvmDeltaOperation::Create(jmpBufType, "buf", Linkage::externalLinkage, "", false, 4));
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
  const auto & callHModRef = modRefSummary->GetSimpleNodeModRef(*callHNode);
  EXPECT_TRUE(callHModRef.getModRefNodes().at(allocaPtgNode));

  // The call to k() should NOT contain a in its Mod/Ref set
  const auto & callKModRef = modRefSummary->GetSimpleNodeModRef(*callKNode);
  EXPECT_FALSE(callKModRef.getModRefNodes().count(allocaPtgNode));

  // The call to opaque() within h() only contains the external memory node,
  // since the memory node representing a has been compressed into it
  const auto & callOpaqueModRef = modRefSummary->GetSimpleNodeModRef(*callOpaqueNode);
  EXPECT_EQ(callOpaqueModRef.getModRefNodes().size(), 1u);

  // Check the statistics to ensure that the right functions in the call graph were marked
  auto & statistic = *collector.CollectedStatistics().begin();
  // Only k() is not in the same SCC as <external>
  EXPECT_EQ(statistic.GetMeasurementValue<uint64_t>("#CallGraphSccs"), 2u);
  // g(), k() and h() are the only functions within an active setjmp
  EXPECT_EQ(statistic.GetMeasurementValue<uint64_t>("#FunctionsCallingSetjmp"), 1u);
}

TEST(RegionAwareModRefSummarizerTests, TestEscapedFunction)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Creates the RVSDG equivalent of the program
   *
   * void opaque();
   * static int global;
   *
   * void f() {
   *   global = global + 1;
   *   opaque();
   *   return global;
   * }
   *
   * The ModRefSet for the call to opaque() should contain the global variable "global",
   * since f() can be called from external functions.
   */

  LlvmRvsdgModule rvsdgModule(util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto & rootRegion = graph.GetRootRegion();

  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto int32Type = rvsdg::BitType::Create(32);

  const auto opaqueFunctionType = rvsdg::FunctionType::Create(
      { ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  const auto fFunctionType = rvsdg::FunctionType::Create(
      { ioStateType, memoryStateType },
      { int32Type, ioStateType, memoryStateType });

  auto & opaqueImport = LlvmGraphImport::createFunctionImport(
      graph,
      opaqueFunctionType,
      "opaque",
      Linkage::externalLinkage,
      CallingConvention::Default);

  auto & global = *rvsdg::DeltaNode::Create(
      &rootRegion,
      LlvmDeltaOperation::Create(int32Type, "global", Linkage::internalLinkage, "", false, 4));
  global.finalize(IntegerConstantOperation::Create(*global.subregion(), 32, 0).output(0));

  rvsdg::SimpleNode * opaqueCallNode = nullptr;
  auto & fLambdaNode = *rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(fFunctionType, "f", Linkage::externalLinkage));
  {
    const auto arguments = fLambdaNode.GetFunctionArguments();
    auto ioState = arguments.at(0);
    auto memoryState = arguments.at(1);

    const auto globalCtxVar = fLambdaNode.AddContextVar(global.output());
    const auto opaqueCtxVar = fLambdaNode.AddContextVar(opaqueImport);

    const auto loadOutputs =
        LoadNonVolatileOperation::Create(globalCtxVar.inner, { memoryState }, int32Type, 4);
    const auto one = IntegerConstantOperation::Create(*fLambdaNode.subregion(), 32, 1).output(0);
    const auto incrementedGlobal =
        rvsdg::CreateOpNode<IntegerAddOperation>({ loadOutputs[0], one }, 32).output(0);
    const auto storeOutputs = StoreNonVolatileOperation::Create(
        globalCtxVar.inner,
        incrementedGlobal,
        { loadOutputs[1] },
        4);

    const auto opaqueCall =
        CallOperation::Create(opaqueCtxVar.inner, opaqueFunctionType, { ioState, storeOutputs[0] });
    opaqueCallNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*opaqueCall[0]);
    ioState = opaqueCall[0];
    memoryState = opaqueCall[1];

    const auto returnLoadOutputs =
        LoadNonVolatileOperation::Create(globalCtxVar.inner, { memoryState }, int32Type, 4);

    fLambdaNode.finalize({ returnLoadOutputs[0], ioState, returnLoadOutputs[1] });
  }

  rvsdg::GraphExport::Create(*fLambdaNode.output(), "f");

  const auto pointsToGraph = RunAndersen(rvsdgModule);
  const auto modRefSummary = aa::RegionAwareModRefSummarizer::Create(rvsdgModule, *pointsToGraph);

  const auto globalMemoryNode = pointsToGraph->getNodeForDelta(global);
  const auto externalMemoryNode = pointsToGraph->getExternalMemoryNode();
  const util::HashSet expectedMemoryNodes{ globalMemoryNode, externalMemoryNode };

  const auto & opaqueCallModRef = modRefSummary->GetSimpleNodeModRef(*opaqueCallNode);
  ASSERT_TRUE(assertSetContains(
      opaqueCallModRef,
      allWithEffect(expectedMemoryNodes, ModRefEffect::ModRef)));
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
  EXPECT_TRUE(statistics.HasTimer("SimpleAllocasSetTimer"));
  EXPECT_TRUE(statistics.HasTimer("NonReentrantAllocaSetsTimer"));
  EXPECT_TRUE(statistics.HasTimer("AnnotationTimer"));
  EXPECT_TRUE(statistics.HasTimer("SolvingTimer"));
  EXPECT_TRUE(statistics.HasTimer("ModRefSetMaterializationTimer"));
}
