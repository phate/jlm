/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/util/Statistics.hpp>

/**
 * A simple test analysis that does nothing else than creating some points-to graph nodes and edges.
 */
class TestAnalysis final : public jlm::llvm::aa::PointsToAnalysis
{
public:
  std::unique_ptr<jlm::llvm::aa::PointsToGraph>
  Analyze(const jlm::rvsdg::RvsdgModule & rvsdgModule, jlm::util::StatisticsCollector &) override
  {
    pointsToGraph_ = jlm::llvm::aa::PointsToGraph::create();

    AnalyzeImports(rvsdgModule.Rvsdg());
    AnalyzeRegion(rvsdgModule.Rvsdg().GetRootRegion());

    return std::move(pointsToGraph_);
  }

  std::unique_ptr<jlm::llvm::aa::PointsToGraph>
  Analyze(const jlm::llvm::RvsdgModule & rvsdgModule)
  {
    jlm::util::StatisticsCollector statisticsCollector;
    return Analyze(rvsdgModule, statisticsCollector);
  }

  static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
  CreateAndAnalyze(const jlm::llvm::RvsdgModule & rvsdgModule)
  {
    TestAnalysis analysis;
    return analysis.Analyze(rvsdgModule);
  }

private:
  void
  AnalyzeRegion(jlm::rvsdg::Region & region)
  {
    using namespace jlm::llvm;

    for (auto & node : region.Nodes())
    {
      if (jlm::rvsdg::is<AllocaOperation>(&node))
      {
        auto simpleNode = jlm::util::assertedCast<jlm::rvsdg::SimpleNode>(&node);
        auto ptgAllocaNode = pointsToGraph_->addNodeForAlloca(*simpleNode, false);
        auto ptgRegisterNode = pointsToGraph_->addNodeForRegisters();
        pointsToGraph_->mapRegisterToNode(*node.output(0), ptgRegisterNode);
        pointsToGraph_->addTarget(ptgRegisterNode, ptgAllocaNode);
      }
      else if (jlm::rvsdg::is<MallocOperation>(&node))
      {
        auto simpleNode = jlm::util::assertedCast<jlm::rvsdg::SimpleNode>(&node);
        auto ptgMallocNode = pointsToGraph_->addNodeForMalloc(*simpleNode, true);
        auto ptgRegisterNode = pointsToGraph_->addNodeForRegisters();
        pointsToGraph_->mapRegisterToNode(*node.output(0), ptgRegisterNode);
        pointsToGraph_->addTarget(ptgRegisterNode, ptgMallocNode);
      }
      else if (auto deltaNode = dynamic_cast<const jlm::rvsdg::DeltaNode *>(&node))
      {
        auto ptgDeltaNode = pointsToGraph_->addNodeForDelta(*deltaNode, true);
        auto ptgRegisterNode = pointsToGraph_->addNodeForRegisters();
        pointsToGraph_->mapRegisterToNode(*node.output(0), ptgRegisterNode);
        pointsToGraph_->addTarget(ptgRegisterNode, ptgDeltaNode);

        AnalyzeRegion(*deltaNode->subregion());
      }
      else if (auto lambdaNode = dynamic_cast<const jlm::rvsdg::LambdaNode *>(&node))
      {
        auto ptgLambdaNode = pointsToGraph_->addNodeForLambda(*lambdaNode, true);
        auto ptgRegisterNode = pointsToGraph_->addNodeForRegisters();
        pointsToGraph_->mapRegisterToNode(*node.output(0), ptgRegisterNode);
        pointsToGraph_->addTarget(ptgRegisterNode, ptgLambdaNode);

        AnalyzeRegion(*lambdaNode->subregion());
      }
    }
  }

  void
  AnalyzeImports(const jlm::rvsdg::Graph & rvsdg)
  {
    using namespace jlm::llvm;

    auto & rootRegion = rvsdg.GetRootRegion();
    for (size_t n = 0; n < rootRegion.narguments(); n++)
    {
      auto & graphImport = *jlm::util::assertedCast<const GraphImport>(rootRegion.argument(n));

      const auto ptgImportNode = pointsToGraph_->addNodeForImport(graphImport, true);
      const auto ptgRegisterNode = pointsToGraph_->addNodeForRegisters();
      pointsToGraph_->mapRegisterToNode(graphImport, ptgRegisterNode);
      pointsToGraph_->addTarget(ptgRegisterNode, ptgImportNode);
    }
  }

  std::unique_ptr<jlm::llvm::aa::PointsToGraph> pointsToGraph_;
};

static void
TestNodeIterators()
{
  // Arrange
  jlm::llvm::AllMemoryNodesTest test;
  auto pointsToGraph = TestAnalysis::CreateAndAnalyze(test.module());

  using NodeIndex = jlm::llvm::aa::PointsToGraph::NodeIndex;
  using NodeKind = jlm::llvm::aa::PointsToGraph::NodeKind;

  // Act and Assert
  assert(pointsToGraph->numImportNodes() == 1);
  for (auto importNode : pointsToGraph->importNodes())
  {
    assert(pointsToGraph->getNodeKind(importNode) == NodeKind::ImportNode);
    assert(&pointsToGraph->getImportForNode(importNode) == &test.GetImportOutput());
  }

  assert(pointsToGraph->numLambdaNodes() == 1);
  for (auto & lambdaNode : pointsToGraph->lambdaNodes())
  {
    assert(pointsToGraph->getNodeKind(lambdaNode) == NodeKind::LambdaNode);
    assert(&pointsToGraph->getLambdaForNode(lambdaNode) == &test.GetLambdaNode());
  }

  assert(pointsToGraph->numDeltaNodes() == 1);
  for (auto & deltaNode : pointsToGraph->deltaNodes())
  {
    assert(pointsToGraph->getNodeKind(deltaNode) == NodeKind::DeltaNode);
    assert(&pointsToGraph->getDeltaForNode(deltaNode) == &test.GetDeltaNode());
  }

  assert(pointsToGraph->numAllocaNodes() == 1);
  for (auto & allocaNode : pointsToGraph->allocaNodes())
  {
    assert(pointsToGraph->getNodeKind(allocaNode) == NodeKind::AllocaNode);
    assert(&pointsToGraph->getAllocaForNode(allocaNode) == &test.GetAllocaNode());
  }

  assert(pointsToGraph->numMallocNodes() == 1);
  for (auto & mallocNode : pointsToGraph->mallocNodes())
  {
    assert(pointsToGraph->getNodeKind(mallocNode) == NodeKind::MallocNode);
    assert(&pointsToGraph->getMallocForNode(mallocNode) == &test.GetMallocNode());
  }

  // Check that each register is seen
  assert(pointsToGraph->numRegisterNodes() == 5);
  jlm::util::HashSet<NodeIndex> seenRegisterNodes;
  for (auto & registerNode : pointsToGraph->registerNodes())
  {
    assert(pointsToGraph->getNodeKind(registerNode) == NodeKind::RegisterNode);
    assert(pointsToGraph->getExplicitTargets(registerNode).Size() == 1);
    seenRegisterNodes.insert(registerNode);
  }
  assert(seenRegisterNodes.Size() == 5);

  const auto ptgImportRegister = pointsToGraph->getNodeForRegister(test.GetImportOutput());
  assert(seenRegisterNodes.Contains(ptgImportRegister));
  const auto ptgLambdaRegister = pointsToGraph->getNodeForRegister(test.GetLambdaOutput());
  assert(seenRegisterNodes.Contains(ptgLambdaRegister));
  const auto ptgDeltaRegister = pointsToGraph->getNodeForRegister(test.GetDeltaOutput());
  assert(seenRegisterNodes.Contains(ptgDeltaRegister));
  const auto ptgAllocaRegister = pointsToGraph->getNodeForRegister(test.GetAllocaOutput());
  assert(seenRegisterNodes.Contains(ptgAllocaRegister));
  const auto ptgMallocRegister = pointsToGraph->getNodeForRegister(test.GetMallocOutput());
  assert(seenRegisterNodes.Contains(ptgMallocRegister));

  // Check that the target of each register node is correct
  const auto ptgImportNode = pointsToGraph->getNodeForImport(test.GetImportOutput());
  assert(pointsToGraph->getExplicitTargets(ptgImportRegister).Contains(ptgImportNode));
  const auto ptgLambdaNode = pointsToGraph->getNodeForLambda(test.GetLambdaNode());
  assert(pointsToGraph->getExplicitTargets(ptgLambdaRegister).Contains(ptgLambdaNode));
  const auto ptgDeltaNode = pointsToGraph->getNodeForDelta(test.GetDeltaNode());
  assert(pointsToGraph->getExplicitTargets(ptgDeltaRegister).Contains(ptgDeltaNode));
  const auto ptgAllocaNode = pointsToGraph->getNodeForAlloca(test.GetAllocaNode());
  assert(pointsToGraph->getExplicitTargets(ptgAllocaRegister).Contains(ptgAllocaNode));
  const auto ptgMallocNode = pointsToGraph->getNodeForMalloc(test.GetMallocNode());
  assert(pointsToGraph->getExplicitTargets(ptgMallocRegister).Contains(ptgMallocNode));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/PointsToGraphTests-TestNodeIterators",
    TestNodeIterators)

static void
TestIsSupergraphOf()
{
  using namespace jlm::llvm::aa;
  auto graph0 = PointsToGraph::create();
  auto graph1 = PointsToGraph::create();

  // Empty graphs are identical, and are both subgraphs of each other
  assert(graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));

  jlm::llvm::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  // Adding an alloca node to only graph0, makes graph1 NOT a subgraph
  const auto alloca0 = graph0->addNodeForAlloca(rvsdg.GetAllocaNode(), false);
  assert(graph0->isSupergraphOf(*graph1));
  assert(!graph1->isSupergraphOf(*graph0));

  // Adding a corresponding alloca node to graph1, they are now equal
  const auto alloca1 = graph1->addNodeForAlloca(rvsdg.GetAllocaNode(), false);
  assert(graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));

  // Adding register0 makes graph1 NOT a subgraph
  const auto register0 = graph0->addNodeForRegisters();
  graph0->mapRegisterToNode(rvsdg.GetAllocaOutput(), register0);
  assert(graph0->isSupergraphOf(*graph1));
  assert(!graph1->isSupergraphOf(*graph0));

  // Adding register1 that covers both the alloca and delta outputs makes graph0 NOT a subgraph
  const auto register1 = graph1->addNodeForRegisters();
  graph1->mapRegisterToNode(rvsdg.GetAllocaOutput(), register1);
  graph1->mapRegisterToNode(rvsdg.GetDeltaOutput(), register1);
  assert(!graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));

  // Adding a deltaRegister0 to make the graphs identical again
  const auto deltaRegister0 = graph0->addNodeForRegisters();
  graph0->mapRegisterToNode(rvsdg.GetDeltaOutput(), deltaRegister0);
  assert(graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));

  // Adding an edge from register0 to the alloca makes graph1 NOT a subgraph
  graph0->addTarget(register0, alloca0);
  assert(graph0->isSupergraphOf(*graph1));
  assert(!graph1->isSupergraphOf(*graph0));

  // By adding an edge from register1 (delta+alloca output), graph0 is now a subgraph of graph1
  graph1->addTarget(register1, alloca1);
  assert(!graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));

  // To make them identical, the both the delta and alloca registers must point to the alloca
  graph0->addTarget(deltaRegister0, alloca0);
  assert(graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));

  // Adding an edge from alloca0 to external that is NOT in graph1
  graph0->markAsTargetsAllExternallyAvailable(alloca0);
  assert(graph0->isSupergraphOf(*graph1));
  assert(!graph1->isSupergraphOf(*graph0));

  // Adding the same edge to alloca1 makes the graphs identical again
  graph1->markAsTargetsAllExternallyAvailable(alloca1);
  assert(graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));

  // Finally test all the other memory node types
  graph0->addNodeForDelta(rvsdg.GetDeltaNode(), false);
  graph1->addNodeForDelta(rvsdg.GetDeltaNode(), false);
  const auto import0 = graph0->addNodeForImport(rvsdg.GetImportOutput(), true);
  const auto import1 = graph1->addNodeForImport(rvsdg.GetImportOutput(), true);
  const auto lambda0 = graph0->addNodeForLambda(rvsdg.GetLambdaNode(), false);
  const auto lambda1 = graph1->addNodeForLambda(rvsdg.GetLambdaNode(), false);
  const auto malloc0 = graph0->addNodeForMalloc(rvsdg.GetMallocNode(), false);
  const auto malloc1 = graph1->addNodeForMalloc(rvsdg.GetMallocNode(), false);
  assert(graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));

  // Add some arbitrary edges between memory nodes
  graph0->addTarget(malloc0, import0);
  assert(graph0->isSupergraphOf(*graph1));
  assert(!graph1->isSupergraphOf(*graph0));

  graph1->addTarget(import1, lambda1);
  assert(!graph0->isSupergraphOf(*graph1));

  // Make them equal again by adding the same edges to the opposite graph
  graph1->addTarget(malloc1, import1);
  graph0->addTarget(import0, lambda0);
  assert(graph0->isSupergraphOf(*graph1));
  assert(graph1->isSupergraphOf(*graph0));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/PointsToGraphTests-TestIsSupergraphOf",
    TestIsSupergraphOf)

static void
testMemoryNodeSize()
{
  using namespace jlm::llvm;

  {
    // Arrange
    jlm::llvm::DeltaTest3 test;
    test.InitializeTest();

    auto ptg = aa::PointsToGraph::create();
    const auto deltaG1 = ptg->addNodeForDelta(test.DeltaG1(), false);
    const auto deltaG2 = ptg->addNodeForDelta(test.DeltaG2(), false);
    const auto f = ptg->addNodeForLambda(test.LambdaF(), false);

    // Assert
    assert(ptg->tryGetNodeSize(deltaG1) == 4);
    assert(ptg->tryGetNodeSize(deltaG2) == 8);
    assert(ptg->tryGetNodeSize(f) == 0);
  }

  {
    // Arrange 2
    jlm::llvm::StoreTest1 test;
    test.InitializeTest();

    auto ptg = aa::PointsToGraph::create();
    const auto allocaD = ptg->addNodeForAlloca(*test.alloca_d, false);
    const auto allocaC = ptg->addNodeForAlloca(*test.alloca_c, false);

    // Assert 2
    assert(ptg->tryGetNodeSize(allocaD) == 4);
    assert(ptg->tryGetNodeSize(allocaC) == 8); // Pointers are 8 bytes
  }

  {
    // Arrange 3
    jlm::llvm::AllMemoryNodesTest test;
    test.InitializeTest();

    auto ptg = aa::PointsToGraph::create();
    const auto allocaNode = ptg->addNodeForAlloca(test.GetAllocaNode(), false);
    const auto mallocNode = ptg->addNodeForMalloc(test.GetMallocNode(), false);
    const auto deltaNode = ptg->addNodeForDelta(test.GetDeltaNode(), true);
    const auto lambdaNode = ptg->addNodeForLambda(test.GetLambdaNode(), true);
    const auto importNode = ptg->addNodeForImport(test.GetImportOutput(), true);

    // Assert 3
    assert(ptg->tryGetNodeSize(allocaNode) == 8);
    assert(ptg->tryGetNodeSize(mallocNode) == 4);
    assert(ptg->tryGetNodeSize(deltaNode) == 8);
    assert(ptg->tryGetNodeSize(importNode) == 4);
    // Function nodes have size 0
    assert(ptg->tryGetNodeSize(lambdaNode) == 0);
  }
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/PointsToGraphTests-testMemoryNodeSize",
    testMemoryNodeSize)

static void
testIsMemoryNodeConstant()
{
  using namespace jlm::llvm;

  {
    // Arrange
    jlm::llvm::AllMemoryNodesTest test;
    test.InitializeTest();

    auto ptg = aa::PointsToGraph::create();
    const auto allocaNode = ptg->addNodeForAlloca(test.GetAllocaNode(), false);
    const auto mallocNode = ptg->addNodeForMalloc(test.GetMallocNode(), false);
    const auto deltaNode = ptg->addNodeForDelta(test.GetDeltaNode(), true);
    const auto lambdaNode = ptg->addNodeForLambda(test.GetLambdaNode(), true);
    const auto importNode = ptg->addNodeForImport(test.GetImportOutput(), true);

    // Assert
    assert(!ptg->isNodeConstant(allocaNode));
    assert(!ptg->isNodeConstant(mallocNode));
    assert(!ptg->isNodeConstant(deltaNode));
    assert(!ptg->isNodeConstant(importNode));
    // Functions are always constant
    assert(ptg->isNodeConstant(lambdaNode));
  }

  {
    // Arrange 2
    jlm::rvsdg::Graph graph;
    const auto intType = jlm::rvsdg::BitType::Create(32);
    const auto pointerType = jlm::llvm::PointerType::Create();

    auto & constImport =
        GraphImport::Create(graph, intType, pointerType, "test", Linkage::externalLinkage, true);
    auto & nonConstImport =
        GraphImport::Create(graph, intType, pointerType, "test", Linkage::externalLinkage, false);

    auto & constDelta = *jlm::rvsdg::DeltaNode::Create(
        &graph.GetRootRegion(),
        DeltaOperation::Create(intType, "constGlobal", Linkage::internalLinkage, "data", true));
    const auto & int2 = IntegerConstantOperation::Create(*constDelta.subregion(), 32, 2);
    constDelta.finalize(int2.output(0));

    auto & nonConstDelta = *jlm::rvsdg::DeltaNode::Create(
        &graph.GetRootRegion(),
        DeltaOperation::Create(intType, "global", Linkage::internalLinkage, "data", false));
    const auto & int8 = IntegerConstantOperation::Create(*nonConstDelta.subregion(), 32, 8);
    nonConstDelta.finalize(int8.output(0));

    auto ptg = aa::PointsToGraph::create();
    const auto constImportMemoryNode = ptg->addNodeForImport(constImport, true);
    const auto nonConstImportMemoryNode = ptg->addNodeForImport(nonConstImport, true);

    const auto & constDeltaMemoryNode = ptg->addNodeForDelta(constDelta, false);
    const auto & nonConstDeltaMemoryNode = ptg->addNodeForDelta(nonConstDelta, false);

    // Assert
    assert(ptg->isNodeConstant(constImportMemoryNode));
    assert(!ptg->isNodeConstant(nonConstImportMemoryNode));
    assert(ptg->isNodeConstant(constDeltaMemoryNode));
    assert(!ptg->isNodeConstant(nonConstDeltaMemoryNode));
  }
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/PointsToGraphTests-testIsMemoryNodeConstant",
    testIsMemoryNodeConstant)
