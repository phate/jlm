/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

class NodeReduction::Statistics final : public util::Statistics
{
public:
  ~Statistics() noexcept override = default;

  explicit Statistics(const util::filepath & sourceFile)
      : util::Statistics(Statistics::Id::ReduceNodes, sourceFile)
  {}

  void
  Start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(graph.root()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(graph.root()));
    AddTimer(Label::Timer).start();
  }

  void
  End(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(graph.root()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(graph.root()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

NodeReduction::~NodeReduction() noexcept = default;

NodeReduction::NodeReduction() = default;

void
NodeReduction::run(RvsdgModule & rvsdgModule)
{
  util::StatisticsCollector statisticsCollector;
  run(rvsdgModule, statisticsCollector);
}

void
NodeReduction::run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
{
  auto & graph = rvsdgModule.Rvsdg();

  auto statistics = Statistics::Create(rvsdgModule.SourceFileName());
  statistics->Start(graph);

  ReduceNodesInRegion(*graph.root());

  statistics->End(graph);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
NodeReduction::ReduceNodesInRegion(rvsdg::Region & region)
{
  bool reductionPerformed;
  do
  {
    reductionPerformed = false;

    for (auto node : jlm::rvsdg::topdown_traverser(&region))
    {
      if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
      {
        reductionPerformed |= ReduceStructuralNode(*structuralNode);
      }
      else if (rvsdg::is<rvsdg::SimpleOperation>(node))
      {
        reductionPerformed |= ReduceSimpleNode(*node);
      }
      else
      {
        JLM_UNREACHABLE("Unhandled node type.");
      }
    }

    if (reductionPerformed)
    {
      // Let's remove all dead nodes in this region to avoid reductions on
      // dead nodes in the next iteration.
      region.prune(false);
    }

  } while (reductionPerformed);
}

bool
NodeReduction::ReduceStructuralNode(rvsdg::StructuralNode & structuralNode)
{
  bool reductionPerformed = false;

  // Reduce structural nodes
  if (is<rvsdg::GammaOperation>(&structuralNode))
  {
    reductionPerformed |= ReduceGammaNode(structuralNode);
  }

  // Reduce all nodes in the subregions
  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    auto subregion = structuralNode.subregion(n);
    ReduceNodesInRegion(*subregion);
  }

  return reductionPerformed;
}

bool
NodeReduction::ReduceGammaNode(rvsdg::StructuralNode & gammaNode)
{
  JLM_ASSERT(is<rvsdg::GammaOperation>(&gammaNode));

  // FIXME: We can not apply the reduction below due to a bug. See github issue #303
  // rvsdg::ReduceGammaControlConstant

  return rvsdg::ReduceGammaWithStaticallyKnownPredicate(gammaNode);
}

bool
NodeReduction::ReduceSimpleNode(rvsdg::Node & simpleNode)
{
  if (is<LoadNonVolatileOperation>(&simpleNode))
  {
    return ReduceLoadNode(simpleNode);
  }
  else if (is<StoreNonVolatileOperation>(&simpleNode))
  {
    return ReduceStoreNode(simpleNode);
  }
  else if (is<rvsdg::unary_op>(&simpleNode))
  {
    // FIXME: handle the unary node
    // See github issue #304
  }
  else if (is<rvsdg::binary_op>(&simpleNode))
  {
    ReduceBinaryNode(simpleNode);
  }

  return false;
}

bool
NodeReduction::ReduceLoadNode(rvsdg::Node & simpleNode)
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(&simpleNode));

  return rvsdg::ReduceNode<LoadNonVolatileOperation>(NormalizeLoadNode, simpleNode);
}

bool
NodeReduction::ReduceStoreNode(rvsdg::Node & simpleNode)
{
  JLM_ASSERT(is<StoreNonVolatileOperation>(&simpleNode));

  return rvsdg::ReduceNode<StoreNonVolatileOperation>(NormalizeStoreNode, simpleNode);
}

bool
NodeReduction::ReduceBinaryNode(rvsdg::Node & simpleNode)
{
  JLM_ASSERT(is<rvsdg::binary_op>(&simpleNode));

  return rvsdg::ReduceNode<rvsdg::binary_op>(rvsdg::NormalizeBinaryOperation, simpleNode);
}

std::optional<std::vector<rvsdg::output *>>
NodeReduction::NormalizeLoadNode(
    const jlm::llvm::LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<LoadNonVolatileOperation>> loadNodeNormalizations(
      { NormalizeLoadMux,
        NormalizeLoadStore,
        NormalizeLoadAlloca,
        NormalizeLoadDuplicateState,
        NormalizeLoadStoreState,
        NormalizeLoadLoadState });

  return rvsdg::NormalizeSequence<LoadNonVolatileOperation>(
      loadNodeNormalizations,
      operation,
      operands);
}

std::optional<std::vector<rvsdg::output *>>
NodeReduction::NormalizeStoreNode(
    const jlm::llvm::StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<StoreNonVolatileOperation>> storeNodeNormalizations(
      { NormalizeStoreMux,
        NormalizeStoreStore,
        NormalizeStoreAlloca,
        NormalizeStoreDuplicateState });

  return rvsdg::NormalizeSequence<StoreNonVolatileOperation>(
      storeNodeNormalizations,
      operation,
      operands);
}

}
