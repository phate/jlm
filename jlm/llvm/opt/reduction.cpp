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

void
NodeReduction::Statistics::Start(const rvsdg::Graph & graph) noexcept
{
  AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
  AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
  AddTimer(Label::Timer).start();
}

void
NodeReduction::Statistics::End(const rvsdg::Graph & graph) noexcept
{
  AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
  AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
  GetTimer(Label::Timer).stop();
}

bool
NodeReduction::Statistics::AddIteration(const rvsdg::Region & region, size_t numIterations)
{
  const auto it = NumIterations_.find(&region);
  NumIterations_[&region] = numIterations;
  return it != NumIterations_.end();
}

std::optional<size_t>
NodeReduction::Statistics::GetNumIterations(const rvsdg::Region & region) const noexcept
{
  if (const auto it = NumIterations_.find(&region); it != NumIterations_.end())
  {
    return it->second;
  }

  return std::nullopt;
}

NodeReduction::~NodeReduction() noexcept = default;

NodeReduction::NodeReduction() = default;

void
NodeReduction::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  const auto & graph = rvsdgModule.Rvsdg();

  Statistics_ = Statistics::Create(rvsdgModule.SourceFilePath().value());
  Statistics_->Start(graph);

  ReduceNodesInRegion(graph.GetRootRegion());

  Statistics_->End(graph);
  statisticsCollector.CollectDemandedStatistics(std::move(Statistics_));
}

void
NodeReduction::ReduceNodesInRegion(rvsdg::Region & region)
{
  bool reductionPerformed;
  size_t numIterations = 0;
  do
  {
    numIterations++;
    reductionPerformed = false;

    for (const auto node : rvsdg::TopDownTraverser(&region))
    {
      if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
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

  Statistics_->AddIteration(region, numIterations);
}

bool
NodeReduction::ReduceStructuralNode(rvsdg::StructuralNode & structuralNode)
{
  bool reductionPerformed = false;

  // Reduce structural nodes
  if (dynamic_cast<const rvsdg::GammaNode *>(&structuralNode))
  {
    reductionPerformed |= ReduceGammaNode(structuralNode);
  }

  if (reductionPerformed)
  {
    // We can not go through the subregions as the structural node might already have been removed.
    return true;
  }

  // Reduce all nodes in the subregions
  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    const auto subregion = structuralNode.subregion(n);
    ReduceNodesInRegion(*subregion);
  }

  return false;
}

bool
NodeReduction::ReduceGammaNode(rvsdg::StructuralNode & gammaNode)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::GammaNode *>(&gammaNode));

  // FIXME: We can not apply the reduction below due to a bug. See github issue #303
  // rvsdg::ReduceGammaControlConstant

  return ReduceGammaWithStaticallyKnownPredicate(gammaNode);
}

bool
NodeReduction::ReduceSimpleNode(rvsdg::Node & simpleNode)
{
  if (is<LoadNonVolatileOperation>(&simpleNode))
  {
    return ReduceLoadNode(simpleNode);
  }
  if (is<StoreNonVolatileOperation>(&simpleNode))
  {
    return ReduceStoreNode(simpleNode);
  }
  if (is<rvsdg::UnaryOperation>(&simpleNode))
  {
    // FIXME: handle the unary node
    // See github issue #304
    return false;
  }
  if (is<rvsdg::BinaryOperation>(&simpleNode))
  {
    return ReduceBinaryNode(simpleNode);
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
  JLM_ASSERT(is<rvsdg::BinaryOperation>(&simpleNode));

  return rvsdg::ReduceNode<rvsdg::BinaryOperation>(rvsdg::NormalizeBinaryOperation, simpleNode);
}

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeLoadNode(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
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

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeStoreNode(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
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
