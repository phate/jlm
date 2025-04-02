/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/llvm/opt/alias-analyses/TopDownMemoryNodeEliminator.hpp>

template<class Test, class Analysis, class Provider>
static void
ValidateTest(std::function<void(const Test &, const jlm::llvm::aa::MemoryNodeProvisioning &)>
                 validateProvisioning)
{
  static_assert(
      std::is_base_of<jlm::tests::RvsdgTest, Test>::value,
      "Test should be derived from RvsdgTest class.");

  static_assert(
      std::is_base_of_v<jlm::llvm::aa::PointsToAnalysis, Analysis>,
      "Analysis should be derived from PointsToAnalysis class.");

  static_assert(
      std::is_base_of<jlm::llvm::aa::MemoryNodeProvider, Provider>::value,
      "Provider should be derived from MemoryNodeProvider class.");

  Test test;
  auto & rvsdgModule = test.module();

  Analysis aliasAnalysis;
  auto pointsToGraph = aliasAnalysis.Analyze(rvsdgModule);
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  auto seedProvisioning = Provider::Create(rvsdgModule, *pointsToGraph);

  auto provisioning = jlm::llvm::aa::TopDownMemoryNodeEliminator::CreateAndEliminate(
      test.module(),
      *seedProvisioning);

  validateProvisioning(test, *provisioning);
}

static void
ValidateStoreTest1SteensgaardAgnostic(
    const jlm::tests::StoreTest1 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaMemoryNode, &externalMemoryNode });

  auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda);
  assert(lambdaEntryNodes == expectedMemoryNodes);

  auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda);
  assert(lambdaExitNodes == expectedMemoryNodes);
}

static void
ValidateStoreTest2SteensgaardAgnostic(
    const jlm::tests::StoreTest2 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaMemoryNode, &externalMemoryNode });

  auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda);
  assert(lambdaEntryNodes == expectedMemoryNodes);

  auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda);
  assert(lambdaExitNodes == expectedMemoryNodes);
}

static void
ValidateLoadTest1SteensgaardAgnostic(
    const jlm::tests::LoadTest1 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaMemoryNode, &externalMemoryNode });

  auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda);
  assert(lambdaEntryNodes == expectedMemoryNodes);

  auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda);
  assert(lambdaExitNodes == expectedMemoryNodes);
}

static void
ValidateLoadTest2SteensgaardAgnostic(
    const jlm::tests::LoadTest2 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaMemoryNode, &externalMemoryNode });

  auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda);
  assert(lambdaEntryNodes == expectedMemoryNodes);

  auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda);
  assert(lambdaExitNodes == expectedMemoryNodes);
}

static void
ValidateLoadFromUndefTestSteensgaardAgnostic(
    const jlm::tests::LoadFromUndefTest & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(test.Lambda());
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaMemoryNode, &externalMemoryNode });

  auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.Lambda());
  assert(lambdaEntryNodes == expectedMemoryNodes);

  auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.Lambda());
  assert(lambdaExitNodes == expectedMemoryNodes);
}

static void
ValidateCallTest1SteensgaardAgnostic(
    const jlm::tests::CallTest1 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & allocaXMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_x);
  auto & allocaYMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_y);
  auto & allocaZMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca_z);
  auto & lambdaFMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda_f);
  auto & lambdaGMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda_g);
  auto & lambdaHMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda_h);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  // Validate function f
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &allocaXMemoryNode,
          &allocaYMemoryNode,
          &allocaZMemoryNode,
          &lambdaFMemoryNode,
          &lambdaGMemoryNode,
          &lambdaHMemoryNode,
          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_f);
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_f);
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function g
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &allocaXMemoryNode,
          &allocaYMemoryNode,
          &allocaZMemoryNode,
          &lambdaFMemoryNode,
          &lambdaGMemoryNode,
          &lambdaHMemoryNode,
          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_g);
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_g);
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function h
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaFMemoryNode, &lambdaGMemoryNode, &lambdaHMemoryNode, &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_h);
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_h);
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate call to f
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &allocaXMemoryNode,
          &allocaYMemoryNode,
          &allocaZMemoryNode,
          &lambdaFMemoryNode,
          &lambdaGMemoryNode,
          &lambdaHMemoryNode,
          &externalMemoryNode });

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.CallF());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.CallF());
    assert(callExitNodes == expectedMemoryNodes);
  }

  // Validate call to g
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &allocaXMemoryNode,
          &allocaYMemoryNode,
          &allocaZMemoryNode,
          &lambdaFMemoryNode,
          &lambdaGMemoryNode,
          &lambdaHMemoryNode,
          &externalMemoryNode });

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.CallG());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.CallG());
    assert(callExitNodes == expectedMemoryNodes);
  }
}

static void
ValidateIndirectCallTest1SteensgaardAgnostic(
    const jlm::tests::IndirectCallTest1 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaFourMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaFour());
  auto & lambdaThreeMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaThree());
  auto & lambdaIndCallMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaIndcall());
  auto & lambdaTestMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaTest());
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaFourMemoryNode,
        &lambdaThreeMemoryNode,
        &lambdaIndCallMemoryNode,
        &lambdaTestMemoryNode,
        &externalMemoryNode });

  // Validate function four
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaFour());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaFour());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function three
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaThree());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaThree());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function indcall
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaIndcall());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaIndcall());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function test
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaIndcall());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaIndcall());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate call to indcall with four
  {
    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.CallFour());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.CallFour());
    assert(callExitNodes == expectedMemoryNodes);
  }

  // Validate call to indcall with three
  {
    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.CallThree());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.CallThree());
    assert(callExitNodes == expectedMemoryNodes);
  }

  // Validate indirect call
  {
    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.CallIndcall());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.CallIndcall());
    assert(callExitNodes == expectedMemoryNodes);
  }
}

static void
ValidateIndirectCallTest2SteensgaardAgnostic(
    const jlm::tests::IndirectCallTest2 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaThreeMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaThree());
  auto & lambdaFourMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaFour());
  auto & lambdaIMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaI());
  auto & lambdaXMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaX());
  auto & lambdaYMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaY());
  auto & lambdaTestMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaTest());
  auto & lambdaTest2MemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaTest2());

  auto & deltaG1MemoryNode = pointsToGraph.GetDeltaNode(test.GetDeltaG1());
  auto & deltaG2MemoryNode = pointsToGraph.GetDeltaNode(test.GetDeltaG2());

  auto & allocaPxMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPx());
  auto & allocaPyMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPy());
  auto & allocaPzMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaPz());

  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  // Validate function test2
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaTest2());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaTest2());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function test
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaTest());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaTest());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function y
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,

          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaY());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaY());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function x
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaX());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaX());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function i
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaI());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaI());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function four
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaFour());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaFour());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function three
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaThree());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaThree());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate indirect call
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetIndirectCall());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetIndirectCall());
    assert(callExitNodes == expectedMemoryNodes);
  }

  // Validate call to i from x
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetCallIWithThree());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetCallIWithThree());
    assert(callExitNodes == expectedMemoryNodes);
  }

  // Validate call to i from y
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetCallIWithFour());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetCallIWithFour());
    assert(callExitNodes == expectedMemoryNodes);
  }

  // Validate call to x from test
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetTestCallX());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetTestCallX());
    assert(callExitNodes == expectedMemoryNodes);
  }

  // Validate call to y from test
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,

          &externalMemoryNode });

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetCallY());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetCallY());
    assert(callExitNodes == expectedMemoryNodes);
  }

  // Validate call to x from test2
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaThreeMemoryNode,
          &lambdaFourMemoryNode,
          &lambdaIMemoryNode,
          &lambdaXMemoryNode,
          &lambdaYMemoryNode,
          &lambdaTestMemoryNode,
          &lambdaTest2MemoryNode,

          &deltaG1MemoryNode,
          &deltaG2MemoryNode,

          &allocaPxMemoryNode,
          &allocaPyMemoryNode,
          &allocaPzMemoryNode,

          &externalMemoryNode });

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetTest2CallX());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetTest2CallX());
    assert(callExitNodes == expectedMemoryNodes);
  }
}

static void
ValidateGammaTestSteensgaardAgnostic(
    const jlm::tests::GammaTest & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaMemoryNode, &externalMemoryNode });

  auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda);
  assert(lambdaEntryNodes == expectedMemoryNodes);

  auto gammaEntryNodes = provisioning.GetGammaEntryNodes(*test.gamma);
  assert(gammaEntryNodes == expectedMemoryNodes);

  for (size_t n = 0; n < test.gamma->nsubregions(); n++)
  {
    auto & subregion = *test.gamma->subregion(n);

    auto & subregionEntryNodes = provisioning.GetRegionEntryNodes(subregion);
    assert(subregionEntryNodes == expectedMemoryNodes);

    auto & subregionExitNodes = provisioning.GetRegionExitNodes(subregion);
    assert(subregionExitNodes == expectedMemoryNodes);
  }

  auto gammaExitNodes = provisioning.GetGammaExitNodes(*test.gamma);
  assert(gammaExitNodes == expectedMemoryNodes);

  auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda);
  assert(lambdaExitNodes == expectedMemoryNodes);
}

static void
ValidateGammaTest2SteensgaardAgnostic(
    const jlm::tests::GammaTest2 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & allocaXFromGMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaXFromG());
  auto & allocaYFromGMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaYFromG());
  auto & allocaXFromHMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaXFromH());
  auto & allocaYFromHMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaYFromH());
  auto & allocaZMemoryNode = pointsToGraph.GetAllocaNode(test.GetAllocaZ());
  auto & lambdaFMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaF());
  auto & lambdaGMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaG());
  auto & lambdaHMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaH());
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *>
      expectedLambdaGHEntryExitMemoryNodes(
          { &lambdaFMemoryNode, &lambdaGMemoryNode, &lambdaHMemoryNode, &externalMemoryNode });

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *>
      expectedLambdaFEntryExitMemoryNodes({ &allocaXFromGMemoryNode,
                                            &allocaYFromGMemoryNode,
                                            &allocaXFromHMemoryNode,
                                            &allocaYFromHMemoryNode,
                                            &lambdaFMemoryNode,
                                            &lambdaGMemoryNode,
                                            &lambdaHMemoryNode,
                                            &externalMemoryNode });

  // Validate g
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaG());
    assert(lambdaEntryNodes == expectedLambdaGHEntryExitMemoryNodes);

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetCallFromG());
    assert(callEntryNodes == expectedLambdaFEntryExitMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetCallFromG());
    assert(callExitNodes == expectedLambdaFEntryExitMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaG());
    assert(lambdaExitNodes == expectedLambdaGHEntryExitMemoryNodes);
  }

  // Validate h
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaH());
    assert(lambdaEntryNodes == expectedLambdaGHEntryExitMemoryNodes);

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetCallFromH());
    assert(callEntryNodes == expectedLambdaFEntryExitMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetCallFromH());
    assert(callExitNodes == expectedLambdaFEntryExitMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaH());
    assert(lambdaExitNodes == expectedLambdaGHEntryExitMemoryNodes);
  }

  // Validate f
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedGammaMemoryNodes(
        { &allocaZMemoryNode,
          &allocaXFromGMemoryNode,
          &allocaYFromGMemoryNode,
          &allocaXFromHMemoryNode,
          &allocaYFromHMemoryNode,
          &lambdaFMemoryNode,
          &lambdaGMemoryNode,
          &lambdaHMemoryNode,
          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaF());
    assert(lambdaEntryNodes == expectedLambdaFEntryExitMemoryNodes);

    auto gammaEntryNodes = provisioning.GetGammaEntryNodes(test.GetGamma());
    assert(gammaEntryNodes == expectedGammaMemoryNodes);

    for (size_t n = 0; n < test.GetGamma().nsubregions(); n++)
    {
      auto & subregion = *test.GetGamma().subregion(n);

      auto & subregionEntryNodes = provisioning.GetRegionEntryNodes(subregion);
      assert(subregionEntryNodes == expectedGammaMemoryNodes);

      auto & subregionExitNodes = provisioning.GetRegionExitNodes(subregion);
      assert(subregionExitNodes == expectedGammaMemoryNodes);
    }

    auto gammaExitNodes = provisioning.GetGammaExitNodes(test.GetGamma());
    assert(gammaExitNodes == expectedGammaMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaF());
    assert(lambdaExitNodes == expectedLambdaFEntryExitMemoryNodes);
  }
}

static void
ValidateThetaTestSteensgaardAgnostic(
    const jlm::tests::ThetaTest & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaMemoryNode, &externalMemoryNode });

  auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda);
  assert(lambdaEntryNodes == expectedMemoryNodes);

  auto & thetaEntryExitNodes = provisioning.GetThetaEntryExitNodes(*test.theta);
  assert(thetaEntryExitNodes == expectedMemoryNodes);

  auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda);
  assert(lambdaExitNodes == expectedMemoryNodes);
}

static void
ValidatePhiTest1SteensgaardAgnostic(
    const jlm::tests::PhiTest1 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaFibMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda_fib);
  auto & lambdaTestMemoryNode = pointsToGraph.GetLambdaNode(*test.lambda_test);
  auto & allocaMemoryNode = pointsToGraph.GetAllocaNode(*test.alloca);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  // validate function fib()
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
        { &lambdaFibMemoryNode, &lambdaTestMemoryNode, &allocaMemoryNode, &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_fib);
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto gammaEntryNodes = provisioning.GetGammaEntryNodes(*test.gamma);
    assert(gammaEntryNodes == expectedMemoryNodes);

    auto & callFib1EntryNodes = provisioning.GetCallEntryNodes(test.CallFibm1());
    assert(callFib1EntryNodes == expectedMemoryNodes);

    auto & callFib1ExitNodes = provisioning.GetCallExitNodes(test.CallFibm1());
    assert(callFib1ExitNodes == expectedMemoryNodes);

    auto & callFib2EntryNodes = provisioning.GetCallEntryNodes(test.CallFibm2());
    assert(callFib2EntryNodes == expectedMemoryNodes);

    auto & callFib2ExitNodes = provisioning.GetCallExitNodes(test.CallFibm2());
    assert(callFib2ExitNodes == expectedMemoryNodes);

    auto gammaExitNodes = provisioning.GetGammaExitNodes(*test.gamma);
    assert(gammaExitNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_fib);
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // validate function test()
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedLambdaMemoryNodes(
        { &lambdaFibMemoryNode, &lambdaTestMemoryNode, &externalMemoryNode });

    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedCallMemoryNodes(
        { &lambdaFibMemoryNode, &lambdaTestMemoryNode, &allocaMemoryNode, &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.lambda_test);
    assert(lambdaEntryNodes == expectedLambdaMemoryNodes);

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.CallFib());
    assert(callEntryNodes == expectedCallMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.CallFib());
    assert(callExitNodes == expectedCallMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.lambda_test);
    assert(lambdaExitNodes == expectedLambdaMemoryNodes);
  }
}

static void
ValidatePhiTest2SteensgaardAgnostic(
    const jlm::tests::PhiTest2 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaAMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaA());
  auto & lambdaBMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaB());
  auto & lambdaCMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaC());
  auto & lambdaDMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaD());
  auto & lambdaIMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaI());
  auto & lambdaEightMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaEight());
  auto & lambdaTestMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaTest());
  auto & allocaPaMemoryNode = pointsToGraph.GetAllocaNode(test.GetPaAlloca());
  auto & allocaPbMemoryNode = pointsToGraph.GetAllocaNode(test.GetPbAlloca());
  auto & allocaPcMemoryNode = pointsToGraph.GetAllocaNode(test.GetPcAlloca());
  auto & allocaPdMemoryNode = pointsToGraph.GetAllocaNode(test.GetPdAlloca());
  auto & allocaPTestMemoryNode = pointsToGraph.GetAllocaNode(test.GetPTestAlloca());
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaAMemoryNode,
        &lambdaBMemoryNode,
        &lambdaCMemoryNode,
        &lambdaDMemoryNode,
        &lambdaIMemoryNode,
        &lambdaEightMemoryNode,
        &lambdaTestMemoryNode,
        &allocaPaMemoryNode,
        &allocaPbMemoryNode,
        &allocaPcMemoryNode,
        &allocaPdMemoryNode,
        &allocaPTestMemoryNode,
        &externalMemoryNode });

  // validate function eight()
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaEight());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaEight());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // validate function i()
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaI());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetIndirectCall());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetIndirectCall());
    assert(callExitNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaI());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // validate function a()
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaA());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & callBEntryNodes = provisioning.GetCallEntryNodes(test.GetCallB());
    assert(callBEntryNodes == expectedMemoryNodes);

    auto & callBExitNodes = provisioning.GetCallExitNodes(test.GetCallB());
    assert(callBExitNodes == expectedMemoryNodes);

    auto & callDEntryNodes = provisioning.GetCallEntryNodes(test.GetCallD());
    assert(callDEntryNodes == expectedMemoryNodes);

    auto & callDExitNodes = provisioning.GetCallExitNodes(test.GetCallD());
    assert(callDExitNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaA());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // validate function b()
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaB());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & callIEntryNodes = provisioning.GetCallEntryNodes(test.GetCallI());
    assert(callIEntryNodes == expectedMemoryNodes);

    auto & callIExitNodes = provisioning.GetCallExitNodes(test.GetCallI());
    assert(callIExitNodes == expectedMemoryNodes);

    auto & callCEntryNodes = provisioning.GetCallEntryNodes(test.GetCallC());
    assert(callCEntryNodes == expectedMemoryNodes);

    auto & callCExitNodes = provisioning.GetCallExitNodes(test.GetCallC());
    assert(callCExitNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaB());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // validate function c()
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaC());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetCallAFromC());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetCallAFromC());
    assert(callExitNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaC());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // validate function d()
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaD());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetCallAFromD());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetCallAFromD());
    assert(callExitNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaD());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // validate function test()
  {
    jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedLambdaMemoryNodes(
        { &lambdaAMemoryNode,
          &lambdaBMemoryNode,
          &lambdaCMemoryNode,
          &lambdaDMemoryNode,
          &lambdaIMemoryNode,
          &lambdaEightMemoryNode,
          &lambdaTestMemoryNode,
          &externalMemoryNode });

    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.GetLambdaTest());
    assert(lambdaEntryNodes == expectedLambdaMemoryNodes);

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.GetCallAFromTest());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.GetCallAFromTest());
    assert(callExitNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.GetLambdaTest());
    assert(lambdaExitNodes == expectedLambdaMemoryNodes);
  }
}

static void
ValidateEscapedMemoryTest3SteensgaardAgnostic(
    const jlm::tests::EscapedMemoryTest3 & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaTestMemoryNode = pointsToGraph.GetLambdaNode(*test.LambdaTest);
  auto & deltaGlobalMemoryNode = pointsToGraph.GetDeltaNode(*test.DeltaGlobal);
  auto & importMemoryNode = pointsToGraph.GetImportNode(*test.ImportExternalFunction);
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaTestMemoryNode, &deltaGlobalMemoryNode, &importMemoryNode, &externalMemoryNode });

  auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(*test.LambdaTest);
  assert(lambdaEntryNodes == expectedMemoryNodes);

  auto & externalCallEntryNodes = provisioning.GetCallEntryNodes(*test.CallExternalFunction);
  assert(externalCallEntryNodes == expectedMemoryNodes);

  auto & externalCallExitNodes = provisioning.GetCallExitNodes(*test.CallExternalFunction);
  assert(externalCallExitNodes == expectedMemoryNodes);

  auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(*test.LambdaTest);
  assert(lambdaExitNodes == expectedMemoryNodes);
}

static void
ValidateMemcpyTestSteensgaardAgnostic(
    const jlm::tests::MemcpyTest & test,
    const jlm::llvm::aa::MemoryNodeProvisioning & provisioning)
{
  auto & pointsToGraph = provisioning.GetPointsToGraph();

  auto & lambdaFMemoryNode = pointsToGraph.GetLambdaNode(test.LambdaF());
  auto & lambdaGMemoryNode = pointsToGraph.GetLambdaNode(test.LambdaG());
  auto & globalArrayMemoryNode = pointsToGraph.GetDeltaNode(test.GlobalArray());
  auto & localArrayMemoryNode = pointsToGraph.GetDeltaNode(test.LocalArray());
  auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

  jlm::util::HashSet<const jlm::llvm::aa::PointsToGraph::MemoryNode *> expectedMemoryNodes(
      { &lambdaFMemoryNode,
        &lambdaGMemoryNode,
        &globalArrayMemoryNode,
        &localArrayMemoryNode,
        &externalMemoryNode });

  // Validate function f()
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.LambdaF());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.LambdaF());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }

  // Validate function g()
  {
    auto & lambdaEntryNodes = provisioning.GetLambdaEntryNodes(test.LambdaG());
    assert(lambdaEntryNodes == expectedMemoryNodes);

    auto & callEntryNodes = provisioning.GetCallEntryNodes(test.CallF());
    assert(callEntryNodes == expectedMemoryNodes);

    auto & callExitNodes = provisioning.GetCallExitNodes(test.CallF());
    assert(callExitNodes == expectedMemoryNodes);

    auto & lambdaExitNodes = provisioning.GetLambdaExitNodes(test.LambdaG());
    assert(lambdaExitNodes == expectedMemoryNodes);
  }
}

static void
TestStatistics()
{
  // Arrange
  jlm::tests::LoadTest1 test;

  jlm::util::StatisticsCollectorSettings statisticsCollectorSettings(
      { jlm::util::Statistics::Id::TopDownMemoryNodeEliminator });
  jlm::util::StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  auto pointsToGraph = jlm::llvm::aa::PointsToGraph::Create();
  auto provisioning = jlm::llvm::aa::AgnosticMemoryNodeProvider::Create(
      test.module(),
      *pointsToGraph,
      statisticsCollector);

  // Act
  jlm::llvm::aa::TopDownMemoryNodeEliminator::CreateAndEliminate(
      test.module(),
      *provisioning,
      statisticsCollector);

  // Assert
  assert(statisticsCollector.NumCollectedStatistics() == 1);
}

static int
TestTopDownMemoryNodeEliminator()
{
  using namespace jlm::llvm::aa;

  ValidateTest<jlm::tests::StoreTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateStoreTest1SteensgaardAgnostic);

  ValidateTest<jlm::tests::StoreTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateStoreTest2SteensgaardAgnostic);

  ValidateTest<jlm::tests::LoadTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateLoadTest1SteensgaardAgnostic);

  ValidateTest<jlm::tests::LoadTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateLoadTest2SteensgaardAgnostic);

  ValidateTest<jlm::tests::LoadFromUndefTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateLoadFromUndefTestSteensgaardAgnostic);

  ValidateTest<jlm::tests::CallTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateCallTest1SteensgaardAgnostic);

  ValidateTest<jlm::tests::IndirectCallTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateIndirectCallTest1SteensgaardAgnostic);

  ValidateTest<jlm::tests::IndirectCallTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateIndirectCallTest2SteensgaardAgnostic);

  ValidateTest<jlm::tests::GammaTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateGammaTestSteensgaardAgnostic);

  ValidateTest<jlm::tests::GammaTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateGammaTest2SteensgaardAgnostic);

  ValidateTest<jlm::tests::ThetaTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateThetaTestSteensgaardAgnostic);

  ValidateTest<jlm::tests::PhiTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidatePhiTest1SteensgaardAgnostic);

  ValidateTest<jlm::tests::PhiTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidatePhiTest2SteensgaardAgnostic);

  ValidateTest<jlm::tests::EscapedMemoryTest3, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateEscapedMemoryTest3SteensgaardAgnostic);

  ValidateTest<jlm::tests::MemcpyTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateMemcpyTestSteensgaardAgnostic);

  TestStatistics();

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestTopDownMemoryNodeEliminator",
    TestTopDownMemoryNodeEliminator)
