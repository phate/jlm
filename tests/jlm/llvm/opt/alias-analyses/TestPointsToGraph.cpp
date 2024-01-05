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
  Analyze(
      const jlm::llvm::RvsdgModule & rvsdgModule,
      jlm::util::StatisticsCollector & statisticsCollector) override
  {
    PointsToGraph_ = jlm::llvm::aa::PointsToGraph::Create();

    AnalyzeImports(rvsdgModule.Rvsdg());
    AnalyzeRegion(*rvsdgModule.Rvsdg().root());

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
  AnalyzeRegion(jlm::rvsdg::region & region)
  {
    using namespace jlm::llvm;

    for (auto & node : region.nodes)
    {
      if (jlm::rvsdg::is<alloca_op>(&node))
      {
        auto & allocaNode = aa::PointsToGraph::AllocaNode::Create(*PointsToGraph_, node);
        auto & registerNode =
            aa::PointsToGraph::RegisterSetNode::Create(*PointsToGraph_, { node.output(0) });
        registerNode.AddEdge(allocaNode);
      }
      else if (jlm::rvsdg::is<malloc_op>(&node))
      {
        auto & mallocNode = aa::PointsToGraph::MallocNode::Create(*PointsToGraph_, node);
        auto & registerNode =
            aa::PointsToGraph::RegisterSetNode::Create(*PointsToGraph_, { node.output(0) });
        registerNode.AddEdge(mallocNode);
      }
      else if (auto deltaNode = dynamic_cast<const delta::node *>(&node))
      {
        auto & deltaPtgNode = aa::PointsToGraph::DeltaNode::Create(*PointsToGraph_, *deltaNode);
        auto & registerNode =
            aa::PointsToGraph::RegisterSetNode::Create(*PointsToGraph_, { deltaNode->output() });
        registerNode.AddEdge(deltaPtgNode);

        AnalyzeRegion(*deltaNode->subregion());
      }
      else if (auto lambdaNode = dynamic_cast<const lambda::node *>(&node))
      {
        auto & lambdaPtgNode = aa::PointsToGraph::LambdaNode::Create(*PointsToGraph_, *lambdaNode);
        auto & registerNode =
            aa::PointsToGraph::RegisterSetNode::Create(*PointsToGraph_, { lambdaNode->output() });
        registerNode.AddEdge(lambdaPtgNode);

        AnalyzeRegion(*lambdaNode->subregion());
      }
    }
  }

  void
  AnalyzeImports(const jlm::rvsdg::graph & rvsdg)
  {
    using namespace jlm::llvm;

    auto & rootRegion = *rvsdg.root();
    for (size_t n = 0; n < rootRegion.narguments(); n++)
    {
      auto & argument = *rootRegion.argument(n);

      auto & importNode = aa::PointsToGraph::ImportNode::Create(*PointsToGraph_, argument);
      auto & registerNode =
          aa::PointsToGraph::RegisterSetNode::Create(*PointsToGraph_, { &argument });
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

  assert(pointsToGraph->NumRegisterSetNodes() == 5);
  jlm::util::HashSet<const jlm::rvsdg::output *> expectedRegisters({ &test.GetImportOutput(),
                                                                     &test.GetLambdaOutput(),
                                                                     &test.GetDeltaOutput(),
                                                                     &test.GetAllocaOutput(),
                                                                     &test.GetMallocOutput() });
  for (auto & registerNode : pointsToGraph->RegisterSetNodes())
  {
    for (auto & output : registerNode.GetOutputs().Items())
    {
      assert(expectedRegisters.Contains(output));
    }
  }
  for (auto & registerNode : constPointsToGraph->RegisterSetNodes())
  {
    for (auto & output : registerNode.GetOutputs().Items())
    {
      assert(expectedRegisters.Contains(output));
    }
  }
}

static void
TestRegisterSetNodeIteration()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::tests::StoreTest2 test;
  test.InitializeTest();

  auto pointsToGraph = aa::PointsToGraph::Create();

  jlm::util::HashSet<const jlm::rvsdg::output *> registers(
      { test.alloca_a->output(0), test.alloca_b->output(0) });
  aa::PointsToGraph::RegisterSetNode::Create(*pointsToGraph, registers);

  // Act
  size_t numIteratedRegisterSetNodes = 0;
  for ([[maybe_unused]] auto & registerSetNode : pointsToGraph->RegisterSetNodes())
    numIteratedRegisterSetNodes++;

  // Assert
  assert(numIteratedRegisterSetNodes == pointsToGraph->NumRegisterSetNodes());
}

static int
TestPointsToGraph()
{
  TestNodeIterators();
  TestRegisterSetNodeIteration();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestPointsToGraph", TestPointsToGraph)
