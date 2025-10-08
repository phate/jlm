/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
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

NodeReduction::NodeReduction()
    : Transformation("NodeReduction")
{}

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
  bool reductionPerformed = false;
  size_t numIterations = 0;
  do
  {
    numIterations++;
    reductionPerformed = false;

    for (const auto node : rvsdg::TopDownTraverser(&region))
    {
      MatchTypeOrFail(
          *node,
          [this, &reductionPerformed](rvsdg::StructuralNode & structuralNode)
          {
            reductionPerformed |= ReduceStructuralNode(structuralNode);
          },
          [&reductionPerformed](rvsdg::SimpleNode & simpleNode)
          {
            reductionPerformed |= ReduceSimpleNode(simpleNode);
          });
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
NodeReduction::ReduceSimpleNode(rvsdg::SimpleNode & simpleNode)
{
  if (is<LoadNonVolatileOperation>(&simpleNode))
  {
    return ReduceLoadNode(simpleNode);
  }
  if (is<StoreNonVolatileOperation>(&simpleNode))
  {
    return ReduceStoreNode(simpleNode);
  }
  if (is<MemoryStateMergeOperation>(&simpleNode))
  {
    return ReduceMemoryStateMergeNode(simpleNode);
  }
  if (is<MemoryStateJoinOperation>(&simpleNode))
  {
    return rvsdg::ReduceNode<MemoryStateJoinOperation>(NormalizeMemoryStateJoinNode, simpleNode);
  }
  if (is<MemoryStateSplitOperation>(&simpleNode))
  {
    return ReduceMemoryStateSplitNode(simpleNode);
  }
  if (is<LambdaEntryMemoryStateSplitOperation>(&simpleNode))
  {
    return rvsdg::ReduceNode<LambdaEntryMemoryStateSplitOperation>(
        NormalizeLambdaEntryMemoryStateSplitNode,
        simpleNode);
  }
  if (is<CallExitMemoryStateSplitOperation>(&simpleNode))
  {
    return rvsdg::ReduceNode<CallExitMemoryStateSplitOperation>(
        NormalizeCallExitMemoryStateSplitNode,
        simpleNode);
  }
  if (is<LambdaExitMemoryStateMergeOperation>(&simpleNode))
  {
    return ReduceLambdaExitMemoryStateMergeNode(simpleNode);
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
NodeReduction::ReduceLoadNode(rvsdg::SimpleNode & simpleNode)
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(&simpleNode));

  return rvsdg::ReduceNode<LoadNonVolatileOperation>(NormalizeLoadNode, simpleNode);
}

bool
NodeReduction::ReduceStoreNode(rvsdg::SimpleNode & simpleNode)
{
  JLM_ASSERT(is<StoreNonVolatileOperation>(&simpleNode));

  return rvsdg::ReduceNode<StoreNonVolatileOperation>(NormalizeStoreNode, simpleNode);
}

bool
NodeReduction::ReduceBinaryNode(rvsdg::SimpleNode & simpleNode)
{
  JLM_ASSERT(is<rvsdg::BinaryOperation>(&simpleNode));

  return rvsdg::ReduceNode<rvsdg::BinaryOperation>(rvsdg::NormalizeBinaryOperation, simpleNode);
}

bool
NodeReduction::ReduceMemoryStateMergeNode(rvsdg::SimpleNode & simpleNode)
{
  JLM_ASSERT(is<MemoryStateMergeOperation>(&simpleNode));

  return rvsdg::ReduceNode<MemoryStateMergeOperation>(NormalizeMemoryStateMergeNode, simpleNode);
}

bool
NodeReduction::ReduceMemoryStateSplitNode(rvsdg::SimpleNode & simpleNode)
{
  JLM_ASSERT(is<MemoryStateSplitOperation>(&simpleNode));

  return rvsdg::ReduceNode<MemoryStateSplitOperation>(NormalizeMemoryStateSplitNode, simpleNode);
}

bool
NodeReduction::ReduceLambdaExitMemoryStateMergeNode(rvsdg::SimpleNode & simpleNode)
{
  JLM_ASSERT(is<LambdaExitMemoryStateMergeOperation>(&simpleNode));
  return rvsdg::ReduceNode<LambdaExitMemoryStateMergeOperation>(
      NormalizeLambdaExitMemoryStateMergeNode,
      simpleNode);
}

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeLoadNode(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<LoadNonVolatileOperation>> loadNodeNormalizations(
      { LoadNonVolatileOperation::NormalizeLoadMemoryStateMerge,
        LoadNonVolatileOperation::NormalizeLoadStore,
        LoadNonVolatileOperation::NormalizeLoadAlloca,
        LoadNonVolatileOperation::NormalizeDuplicateStates,
        LoadNonVolatileOperation::NormalizeLoadStoreState,
        LoadNonVolatileOperation::NormalizeLoadLoadState,
        LoadNonVolatileOperation::NormalizeIOBarrierAllocaAddress });

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
      { StoreNonVolatileOperation::NormalizeStoreMux,
        StoreNonVolatileOperation::NormalizeStoreStore,
        StoreNonVolatileOperation::NormalizeStoreAlloca,
        StoreNonVolatileOperation::NormalizeDuplicateStates,
        StoreNonVolatileOperation::NormalizeIOBarrierAllocaAddress,
        StoreNonVolatileOperation::normalizeStoreAllocaSingleUser });

  return rvsdg::NormalizeSequence<StoreNonVolatileOperation>(
      storeNodeNormalizations,
      operation,
      operands);
}

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeMemoryStateMergeNode(
    const MemoryStateMergeOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<MemoryStateMergeOperation>> normalizations(
      { MemoryStateMergeOperation::NormalizeSingleOperand,
        MemoryStateMergeOperation::NormalizeDuplicateOperands,
        MemoryStateMergeOperation::NormalizeNestedMerges,
        MemoryStateMergeOperation::NormalizeMergeSplit });

  return rvsdg::NormalizeSequence<MemoryStateMergeOperation>(normalizations, operation, operands);
}

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeMemoryStateJoinNode(
    const MemoryStateJoinOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<MemoryStateJoinOperation>> normalizations(
      { MemoryStateJoinOperation::NormalizeSingleOperand,
        MemoryStateJoinOperation::NormalizeDuplicateOperands });

  return rvsdg::NormalizeSequence<MemoryStateJoinOperation>(normalizations, operation, operands);
}

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeMemoryStateSplitNode(
    const MemoryStateSplitOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<MemoryStateSplitOperation>> normalizations(
      { MemoryStateSplitOperation::NormalizeSingleResult,
        MemoryStateSplitOperation::NormalizeNestedSplits,
        MemoryStateSplitOperation::NormalizeSplitMerge });

  return rvsdg::NormalizeSequence<MemoryStateSplitOperation>(normalizations, operation, operands);
}

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeCallExitMemoryStateSplitNode(
    const CallExitMemoryStateSplitOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<CallExitMemoryStateSplitOperation>> normalizations(
      { CallExitMemoryStateSplitOperation::NormalizeLambdaExitMemoryStateMerge });

  return rvsdg::NormalizeSequence<CallExitMemoryStateSplitOperation>(
      normalizations,
      operation,
      operands);
}

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeLambdaEntryMemoryStateSplitNode(
    const LambdaEntryMemoryStateSplitOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<LambdaEntryMemoryStateSplitOperation>> normalizations(
      { LambdaEntryMemoryStateSplitOperation::NormalizeCallEntryMemoryStateMerge });

  return rvsdg::NormalizeSequence<LambdaEntryMemoryStateSplitOperation>(
      normalizations,
      operation,
      operands);
}

std::optional<std::vector<rvsdg::Output *>>
NodeReduction::NormalizeLambdaExitMemoryStateMergeNode(
    const LambdaExitMemoryStateMergeOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  static std::vector<rvsdg::NodeNormalization<LambdaExitMemoryStateMergeOperation>> normalizations(
      { LambdaExitMemoryStateMergeOperation::NormalizeLoadFromAlloca,
        LambdaExitMemoryStateMergeOperation::NormalizeStoreToAlloca,
        LambdaExitMemoryStateMergeOperation::NormalizeAlloca });

  return rvsdg::NormalizeSequence<LambdaExitMemoryStateMergeOperation>(
      normalizations,
      operation,
      operands);
}

}
