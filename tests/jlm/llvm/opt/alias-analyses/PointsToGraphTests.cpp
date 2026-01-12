/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

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
  Analyze(const jlm::llvm::LlvmRvsdgModule & rvsdgModule)
  {
    jlm::util::StatisticsCollector statisticsCollector;
    return Analyze(rvsdgModule, statisticsCollector);
  }

  static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
  CreateAndAnalyze(const jlm::llvm::LlvmRvsdgModule & rvsdgModule)
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
        pointsToGraph_->mapRegisterToNode(MallocOperation::addressOutput(node), ptgRegisterNode);
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

TEST(PointsToGraphTests, TestNodeIterators)
{
  // Arrange
  jlm::llvm::AllMemoryNodesTest test;
  auto pointsToGraph = TestAnalysis::CreateAndAnalyze(test.module());

  using NodeIndex = jlm::llvm::aa::PointsToGraph::NodeIndex;
  using NodeKind = jlm::llvm::aa::PointsToGraph::NodeKind;

  // Act and Assert
  EXPECT_EQ(pointsToGraph->numImportNodes(), 1u);
  for (auto importNode : pointsToGraph->importNodes())
  {
    EXPECT_EQ(pointsToGraph->getNodeKind(importNode), NodeKind::ImportNode);
    EXPECT_EQ(&pointsToGraph->getImportForNode(importNode), &test.GetImportOutput());
  }

  EXPECT_EQ(pointsToGraph->numLambdaNodes(), 1u);
  for (auto & lambdaNode : pointsToGraph->lambdaNodes())
  {
    EXPECT_EQ(pointsToGraph->getNodeKind(lambdaNode), NodeKind::LambdaNode);
    EXPECT_EQ(&pointsToGraph->getLambdaForNode(lambdaNode), &test.GetLambdaNode());
  }

  EXPECT_EQ(pointsToGraph->numDeltaNodes(), 1u);
  for (auto & deltaNode : pointsToGraph->deltaNodes())
  {
    EXPECT_EQ(pointsToGraph->getNodeKind(deltaNode), NodeKind::DeltaNode);
    EXPECT_EQ(&pointsToGraph->getDeltaForNode(deltaNode), &test.GetDeltaNode());
  }

  EXPECT_EQ(pointsToGraph->numAllocaNodes(), 1u);
  for (auto & allocaNode : pointsToGraph->allocaNodes())
  {
    EXPECT_EQ(pointsToGraph->getNodeKind(allocaNode), NodeKind::AllocaNode);
    EXPECT_EQ(&pointsToGraph->getAllocaForNode(allocaNode), &test.GetAllocaNode());
  }

  EXPECT_EQ(pointsToGraph->numMallocNodes(), 1u);
  for (auto & mallocNode : pointsToGraph->mallocNodes())
  {
    EXPECT_EQ(pointsToGraph->getNodeKind(mallocNode), NodeKind::MallocNode);
    EXPECT_EQ(&pointsToGraph->getMallocForNode(mallocNode), &test.GetMallocNode());
  }

  // Check that each register is seen
  EXPECT_EQ(pointsToGraph->numRegisterNodes(), 5u);
  jlm::util::HashSet<NodeIndex> seenRegisterNodes;
  for (auto & registerNode : pointsToGraph->registerNodes())
  {
    EXPECT_EQ(pointsToGraph->getNodeKind(registerNode), NodeKind::RegisterNode);
    EXPECT_EQ(pointsToGraph->getExplicitTargets(registerNode).Size(), 1u);
    seenRegisterNodes.insert(registerNode);
  }
  EXPECT_EQ(seenRegisterNodes.Size(), 5u);

  const auto ptgImportRegister = pointsToGraph->getNodeForRegister(test.GetImportOutput());
  EXPECT_TRUE(seenRegisterNodes.Contains(ptgImportRegister));
  const auto ptgLambdaRegister = pointsToGraph->getNodeForRegister(test.GetLambdaOutput());
  EXPECT_TRUE(seenRegisterNodes.Contains(ptgLambdaRegister));
  const auto ptgDeltaRegister = pointsToGraph->getNodeForRegister(test.GetDeltaOutput());
  EXPECT_TRUE(seenRegisterNodes.Contains(ptgDeltaRegister));
  const auto ptgAllocaRegister = pointsToGraph->getNodeForRegister(test.GetAllocaOutput());
  EXPECT_TRUE(seenRegisterNodes.Contains(ptgAllocaRegister));
  const auto ptgMallocRegister = pointsToGraph->getNodeForRegister(test.GetMallocOutput());
  EXPECT_TRUE(seenRegisterNodes.Contains(ptgMallocRegister));

  // Check that the target of each register node is correct
  const auto ptgImportNode = pointsToGraph->getNodeForImport(test.GetImportOutput());
  EXPECT_TRUE(pointsToGraph->getExplicitTargets(ptgImportRegister).Contains(ptgImportNode));
  const auto ptgLambdaNode = pointsToGraph->getNodeForLambda(test.GetLambdaNode());
  EXPECT_TRUE(pointsToGraph->getExplicitTargets(ptgLambdaRegister).Contains(ptgLambdaNode));
  const auto ptgDeltaNode = pointsToGraph->getNodeForDelta(test.GetDeltaNode());
  EXPECT_TRUE(pointsToGraph->getExplicitTargets(ptgDeltaRegister).Contains(ptgDeltaNode));
  const auto ptgAllocaNode = pointsToGraph->getNodeForAlloca(test.GetAllocaNode());
  EXPECT_TRUE(pointsToGraph->getExplicitTargets(ptgAllocaRegister).Contains(ptgAllocaNode));
  const auto ptgMallocNode = pointsToGraph->getNodeForMalloc(test.GetMallocNode());
  EXPECT_TRUE(pointsToGraph->getExplicitTargets(ptgMallocRegister).Contains(ptgMallocNode));
}

TEST(PointsToGraphTests, TestIsSupergraphOf)
{
  using namespace jlm::llvm::aa;
  auto graph0 = PointsToGraph::create();
  auto graph1 = PointsToGraph::create();

  // Empty graphs are identical, and are both subgraphs of each other
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));

  jlm::llvm::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  // Adding an alloca node to only graph0, makes graph1 NOT a subgraph
  const auto alloca0 = graph0->addNodeForAlloca(rvsdg.GetAllocaNode(), false);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_FALSE(graph1->isSupergraphOf(*graph0));

  // Adding a corresponding alloca node to graph1, they are now equal
  const auto alloca1 = graph1->addNodeForAlloca(rvsdg.GetAllocaNode(), false);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));

  // Adding register0 makes graph1 NOT a subgraph
  const auto register0 = graph0->addNodeForRegisters();
  graph0->mapRegisterToNode(rvsdg.GetAllocaOutput(), register0);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_FALSE(graph1->isSupergraphOf(*graph0));

  // Adding register1 that covers both the alloca and delta outputs makes graph0 NOT a subgraph
  const auto register1 = graph1->addNodeForRegisters();
  graph1->mapRegisterToNode(rvsdg.GetAllocaOutput(), register1);
  graph1->mapRegisterToNode(rvsdg.GetDeltaOutput(), register1);
  EXPECT_FALSE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));

  // Adding a deltaRegister0 to make the graphs identical again
  const auto deltaRegister0 = graph0->addNodeForRegisters();
  graph0->mapRegisterToNode(rvsdg.GetDeltaOutput(), deltaRegister0);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));

  // Adding an edge from register0 to the alloca makes graph1 NOT a subgraph
  graph0->addTarget(register0, alloca0);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_FALSE(graph1->isSupergraphOf(*graph0));

  // By adding an edge from register1 (delta+alloca output), graph0 is now a subgraph of graph1
  graph1->addTarget(register1, alloca1);
  EXPECT_FALSE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));

  // To make them identical, the both the delta and alloca registers must point to the alloca
  graph0->addTarget(deltaRegister0, alloca0);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));

  // Adding an edge from alloca0 to external that is NOT in graph1
  graph0->markAsTargetsAllExternallyAvailable(alloca0);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_FALSE(graph1->isSupergraphOf(*graph0));

  // Adding the same edge to alloca1 makes the graphs identical again
  graph1->markAsTargetsAllExternallyAvailable(alloca1);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));

  // Finally test all the other memory node types
  graph0->addNodeForDelta(rvsdg.GetDeltaNode(), false);
  graph1->addNodeForDelta(rvsdg.GetDeltaNode(), false);
  const auto import0 = graph0->addNodeForImport(rvsdg.GetImportOutput(), true);
  const auto import1 = graph1->addNodeForImport(rvsdg.GetImportOutput(), true);
  const auto lambda0 = graph0->addNodeForLambda(rvsdg.GetLambdaNode(), false);
  const auto lambda1 = graph1->addNodeForLambda(rvsdg.GetLambdaNode(), false);
  const auto malloc0 = graph0->addNodeForMalloc(rvsdg.GetMallocNode(), false);
  const auto malloc1 = graph1->addNodeForMalloc(rvsdg.GetMallocNode(), false);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));

  // Add some arbitrary edges between memory nodes
  graph0->addTarget(malloc0, import0);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_FALSE(graph1->isSupergraphOf(*graph0));

  graph1->addTarget(import1, lambda1);
  EXPECT_FALSE(graph0->isSupergraphOf(*graph1));

  // Make them equal again by adding the same edges to the opposite graph
  graph1->addTarget(malloc1, import1);
  graph0->addTarget(import0, lambda0);
  EXPECT_TRUE(graph0->isSupergraphOf(*graph1));
  EXPECT_TRUE(graph1->isSupergraphOf(*graph0));
}

TEST(PointsToGraphTests, testMemoryNodeSize)
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
    EXPECT_EQ(ptg->tryGetNodeSize(deltaG1), 4);
    EXPECT_EQ(ptg->tryGetNodeSize(deltaG2), 8);
    EXPECT_EQ(ptg->tryGetNodeSize(f), 0);
  }

  {
    // Arrange 2
    jlm::llvm::StoreTest1 test;
    test.InitializeTest();

    auto ptg = aa::PointsToGraph::create();
    const auto allocaD = ptg->addNodeForAlloca(*test.alloca_d, false);
    const auto allocaC = ptg->addNodeForAlloca(*test.alloca_c, false);

    // Assert 2
    EXPECT_EQ(ptg->tryGetNodeSize(allocaD), 4);
    EXPECT_EQ(ptg->tryGetNodeSize(allocaC), 8); // Pointers are 8 bytes
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
    EXPECT_EQ(ptg->tryGetNodeSize(allocaNode), 8);
    EXPECT_EQ(ptg->tryGetNodeSize(mallocNode), 4);
    EXPECT_EQ(ptg->tryGetNodeSize(deltaNode), 8);
    EXPECT_EQ(ptg->tryGetNodeSize(importNode), 4);
    // Function nodes have size 0
    EXPECT_EQ(ptg->tryGetNodeSize(lambdaNode), 0);
  }
}

TEST(PointsToGraphTests, testIsMemoryNodeConstant)
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
    EXPECT_FALSE(ptg->isNodeConstant(allocaNode));
    EXPECT_FALSE(ptg->isNodeConstant(mallocNode));
    EXPECT_FALSE(ptg->isNodeConstant(deltaNode));
    EXPECT_FALSE(ptg->isNodeConstant(importNode));
    // Functions are always constant
    EXPECT_TRUE(ptg->isNodeConstant(lambdaNode));
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
    EXPECT_TRUE(ptg->isNodeConstant(constImportMemoryNode));
    EXPECT_FALSE(ptg->isNodeConstant(nonConstImportMemoryNode));
    EXPECT_TRUE(ptg->isNodeConstant(constDeltaMemoryNode));
    EXPECT_FALSE(ptg->isNodeConstant(nonConstDeltaMemoryNode));
  }
}
