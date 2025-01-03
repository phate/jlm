/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/Statistics.hpp>

/**
 * A simple test analysis that does nothing else than creating some points-to graph nodes and edges.
 */
class TestAnalysis final : public jlm::llvm::aa::AliasAnalysis
{
public:
  std::unique_ptr<jlm::llvm::aa::PointsToGraph>
  Analyze(const jlm::llvm::RvsdgModule & rvsdgModule, jlm::util::StatisticsCollector &) override
  {
    PointsToGraph_ = jlm::llvm::aa::PointsToGraph::Create();

    AnalyzeImports(rvsdgModule.Rvsdg());
    AnalyzeRegion(rvsdgModule.Rvsdg().GetRootRegion());

    return std::move(PointsToGraph_);
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
      if (jlm::rvsdg::is<alloca_op>(&node))
      {
        auto & allocaNode = aa::PointsToGraph::AllocaNode::Create(*PointsToGraph_, node);
        auto & registerNode =
            aa::PointsToGraph::RegisterNode::Create(*PointsToGraph_, { node.output(0) });
        registerNode.AddEdge(allocaNode);
      }
      else if (jlm::rvsdg::is<malloc_op>(&node))
      {
        auto & mallocNode = aa::PointsToGraph::MallocNode::Create(*PointsToGraph_, node);
        auto & registerNode =
            aa::PointsToGraph::RegisterNode::Create(*PointsToGraph_, { node.output(0) });
        registerNode.AddEdge(mallocNode);
      }
      else if (auto deltaNode = dynamic_cast<const delta::node *>(&node))
      {
        auto & deltaPtgNode = aa::PointsToGraph::DeltaNode::Create(*PointsToGraph_, *deltaNode);
        auto & registerNode =
            aa::PointsToGraph::RegisterNode::Create(*PointsToGraph_, { deltaNode->output() });
        registerNode.AddEdge(deltaPtgNode);

        AnalyzeRegion(*deltaNode->subregion());
      }
      else if (auto lambdaNode = dynamic_cast<const lambda::node *>(&node))
      {
        auto & lambdaPtgNode = aa::PointsToGraph::LambdaNode::Create(*PointsToGraph_, *lambdaNode);
        auto & registerNode =
            aa::PointsToGraph::RegisterNode::Create(*PointsToGraph_, { lambdaNode->output() });
        registerNode.AddEdge(lambdaPtgNode);

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
      auto & graphImport = *jlm::util::AssertedCast<const GraphImport>(rootRegion.argument(n));

      auto & importNode = aa::PointsToGraph::ImportNode::Create(*PointsToGraph_, graphImport);
      auto & registerNode =
          aa::PointsToGraph::RegisterNode::Create(*PointsToGraph_, { &graphImport });
      registerNode.AddEdge(importNode);
    }
  }

  std::unique_ptr<jlm::llvm::aa::PointsToGraph> PointsToGraph_;
};

static void
TestNodeIterators()
{
  // Arrange
  jlm::tests::AllMemoryNodesTest test;
  auto pointsToGraph = TestAnalysis::CreateAndAnalyze(test.module());
  auto constPointsToGraph = static_cast<const jlm::llvm::aa::PointsToGraph *>(pointsToGraph.get());

  // Act and Arrange
  assert(pointsToGraph->NumImportNodes() == 1);
  for (auto & importNode : pointsToGraph->ImportNodes())
  {
    assert(&importNode.GetArgument() == &test.GetImportOutput());
  }
  for (auto & importNode : constPointsToGraph->ImportNodes())
  {
    assert(&importNode.GetArgument() == &test.GetImportOutput());
  }

  assert(pointsToGraph->NumLambdaNodes() == 1);
  for (auto & lambdaNode : pointsToGraph->LambdaNodes())
  {
    assert(&lambdaNode.GetLambdaNode() == &test.GetLambdaNode());
  }
  for (auto & lambdaNode : constPointsToGraph->LambdaNodes())
  {
    assert(&lambdaNode.GetLambdaNode() == &test.GetLambdaNode());
  }

  assert(pointsToGraph->NumDeltaNodes() == 1);
  for (auto & deltaNode : pointsToGraph->DeltaNodes())
  {
    assert(&deltaNode.GetDeltaNode() == &test.GetDeltaNode());
  }
  for (auto & deltaNode : constPointsToGraph->DeltaNodes())
  {
    assert(&deltaNode.GetDeltaNode() == &test.GetDeltaNode());
  }

  assert(pointsToGraph->NumAllocaNodes() == 1);
  for (auto & allocaNode : pointsToGraph->AllocaNodes())
  {
    assert(&allocaNode.GetAllocaNode() == &test.GetAllocaNode());
  }
  for (auto & allocaNode : constPointsToGraph->AllocaNodes())
  {
    assert(&allocaNode.GetAllocaNode() == &test.GetAllocaNode());
  }

  assert(pointsToGraph->NumMallocNodes() == 1);
  for (auto & mallocNode : pointsToGraph->MallocNodes())
  {
    assert(&mallocNode.GetMallocNode() == &test.GetMallocNode());
  }
  for (auto & mallocNode : constPointsToGraph->MallocNodes())
  {
    assert(&mallocNode.GetMallocNode() == &test.GetMallocNode());
  }

  assert(pointsToGraph->NumRegisterNodes() == 5);
  jlm::util::HashSet<const jlm::rvsdg::output *> expectedRegisters({ &test.GetImportOutput(),
                                                                     &test.GetLambdaOutput(),
                                                                     &test.GetDeltaOutput(),
                                                                     &test.GetAllocaOutput(),
                                                                     &test.GetMallocOutput() });
  for (auto & registerNode : pointsToGraph->RegisterNodes())
  {
    for (auto & output : registerNode.GetOutputs().Items())
    {
      assert(expectedRegisters.Contains(output));
    }
  }
  for (auto & registerNode : constPointsToGraph->RegisterNodes())
  {
    for (auto & output : registerNode.GetOutputs().Items())
    {
      assert(expectedRegisters.Contains(output));
    }
  }
}

static void
TestRegisterNodeIteration()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::tests::StoreTest2 test;
  test.InitializeTest();

  auto pointsToGraph = aa::PointsToGraph::Create();

  jlm::util::HashSet<const jlm::rvsdg::output *> registers(
      { test.alloca_a->output(0), test.alloca_b->output(0) });
  aa::PointsToGraph::RegisterNode::Create(*pointsToGraph, registers);

  // Act
  size_t numIteratedRegisterNodes = 0;
  for ([[maybe_unused]] auto & registerNode : pointsToGraph->RegisterNodes())
    numIteratedRegisterNodes++;

  // Assert
  assert(numIteratedRegisterNodes == pointsToGraph->NumRegisterNodes());
}

static void
TestIsSupergraphOf()
{
  using namespace jlm::llvm::aa;
  auto graph0 = PointsToGraph::Create();
  auto graph1 = PointsToGraph::Create();

  // Empty graphs are identical, and are both subgraphs of each other
  assert(graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  // Adding an alloca node to only graph0, makes graph1 NOT a subgraph
  auto & alloca0 = PointsToGraph::AllocaNode::Create(*graph0, rvsdg.GetAllocaNode());
  assert(graph0->IsSupergraphOf(*graph1));
  assert(!graph1->IsSupergraphOf(*graph0));

  // Adding a corresponding alloca node to graph1, they are now equal
  auto & alloca1 = PointsToGraph::AllocaNode::Create(*graph1, rvsdg.GetAllocaNode());
  assert(graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  // Marking alloca0 as escaping, makes graph1 NOT a subgraph
  alloca0.MarkAsModuleEscaping();
  assert(graph0->IsSupergraphOf(*graph1));
  assert(!graph1->IsSupergraphOf(*graph0));

  // Now both alloca0 and alloca1 is marked as escaping
  alloca1.MarkAsModuleEscaping();
  assert(graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  // Adding register0 makes graph1 NOT a subgraph
  auto & register0 = PointsToGraph::RegisterNode::Create(*graph0, { &rvsdg.GetAllocaOutput() });
  assert(graph0->IsSupergraphOf(*graph1));
  assert(!graph1->IsSupergraphOf(*graph0));

  // Adding register1 that covers both the alloca and delta outputs makes graph0 NOT a subgraph
  auto & register1 = PointsToGraph::RegisterNode::Create(
      *graph1,
      { &rvsdg.GetAllocaOutput(), &rvsdg.GetDeltaOutput() });
  assert(!graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  // Adding a deltaRegister0 to make the graphs identical again
  auto & deltaRegister0 = PointsToGraph::RegisterNode::Create(*graph0, { &rvsdg.GetDeltaOutput() });
  assert(graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  // Adding an edge from register0 to the alloca makes graph1 NOT a subgraph
  register0.AddEdge(alloca0);
  assert(graph0->IsSupergraphOf(*graph1));
  assert(!graph1->IsSupergraphOf(*graph0));

  // By adding an edge from register1 (delta+alloca output), graph0 is now a subgraph of graph1
  register1.AddEdge(alloca1);
  assert(!graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  // To make them identical, the both the delta and alloca registers must point to the alloca
  deltaRegister0.AddEdge(alloca0);
  assert(graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  // Adding an edge from alloca0 to external that is NOT in graph1
  alloca0.AddEdge(graph0->GetExternalMemoryNode());
  assert(graph0->IsSupergraphOf(*graph1));
  assert(!graph1->IsSupergraphOf(*graph0));

  // Adding the same edge to alloca1 makes the graphs identical again
  alloca1.AddEdge(graph1->GetExternalMemoryNode());
  assert(graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  // Finally test all the other memory node types
  auto & delta0 = PointsToGraph::DeltaNode::Create(*graph0, rvsdg.GetDeltaNode());
  auto & delta1 = PointsToGraph::DeltaNode::Create(*graph1, rvsdg.GetDeltaNode());
  auto & import0 = PointsToGraph::ImportNode::Create(*graph0, rvsdg.GetImportOutput());
  auto & import1 = PointsToGraph::ImportNode::Create(*graph1, rvsdg.GetImportOutput());
  auto & lambda0 = PointsToGraph::LambdaNode::Create(*graph0, rvsdg.GetLambdaNode());
  auto & lambda1 = PointsToGraph::LambdaNode::Create(*graph1, rvsdg.GetLambdaNode());
  auto & malloc0 = PointsToGraph::MallocNode::Create(*graph0, rvsdg.GetMallocNode());
  auto & malloc1 = PointsToGraph::MallocNode::Create(*graph1, rvsdg.GetMallocNode());
  assert(graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));

  // Add edges to unknown in one graph at a time
  delta0.AddEdge(graph0->GetUnknownMemoryNode());
  assert(graph0->IsSupergraphOf(*graph1));
  assert(!graph1->IsSupergraphOf(*graph0));
  delta1.AddEdge(graph1->GetUnknownMemoryNode());
  assert(graph1->IsSupergraphOf(*graph0));

  // Add some arbitrary edges between memory nodes
  malloc0.AddEdge(import0);
  assert(graph0->IsSupergraphOf(*graph1));
  assert(!graph1->IsSupergraphOf(*graph0));

  import1.AddEdge(lambda1);
  assert(!graph0->IsSupergraphOf(*graph1));

  // Make them equal again by adding the same edges to the opposite graph
  malloc1.AddEdge(import1);
  import0.AddEdge(lambda0);
  assert(graph0->IsSupergraphOf(*graph1));
  assert(graph1->IsSupergraphOf(*graph0));
}

static int
TestPointsToGraph()
{
  TestNodeIterators();
  TestRegisterNodeIteration();
  TestIsSupergraphOf();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestPointsToGraph", TestPointsToGraph)
