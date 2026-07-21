/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/ControlOperations.hpp>
#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/Gamma.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/opt/NodeReduction.hpp>
#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
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

  AddMeasurement(NumRegionsLabel_, getNumRegions());
  AddMeasurement(NumTotalRegionIterationsLabel_, getTotalIterations());
  AddMeasurement(MaxIterationsPerRegionLabel_, getMaxIterationsPerRegion());

  auto & counters = getReductionCounters();
  AddMeasurement("#LoadNonVolatileReductions", counters.numLoadNonVolatileReductions);
  AddMeasurement("#StoreNonVolatileReductions", counters.numStoreNonVolatileReductions);
  AddMeasurement("#MemoryStateMergeReductions", counters.numMemoryStateMergeReductions);
  AddMeasurement("#MemoryStateJoinReductions", counters.numMemoryStateJoinReductions);
  AddMeasurement("#MemoryStateSplitReductions", counters.numMemoryStateSplitReductions);
  AddMeasurement(
      "#LambdaExitMemoryStateMergeReductions",
      counters.numLambdaExitMemoryStateMergeReductions);
  AddMeasurement("#MatchReductions", counters.numMatchReductions);
  AddMeasurement("#SExtReductions", counters.numSExtReductions);
  AddMeasurement("#ZExtReductions", counters.numZExtReductions);
  AddMeasurement("#IntegerEqReductions", counters.numIntegerEqReductions);
  AddMeasurement("#IntegerNeReductions", counters.numIntegerNeReductions);
  AddMeasurement("#IntegerSgeReductions", counters.numIntegerSgeReductions);
  AddMeasurement("#IntegerSgtReductions", counters.numIntegerSgtReductions);
  AddMeasurement("#IntegerSleReductions", counters.numIntegerSleReductions);
  AddMeasurement("#IntegerSltReductions", counters.numIntegerSltReductions);
  AddMeasurement("#IntegerUgeReductions", counters.numIntegerUgeReductions);
  AddMeasurement("#IntegerUgtReductions", counters.numIntegerUgtReductions);
  AddMeasurement("#IntegerUleReductions", counters.numIntegerUleReductions);
  AddMeasurement("#IntegerUltReductions", counters.numIntegerUltReductions);
  AddMeasurement("#PtrCmpReductions", counters.numPtrCmpReductions);
  AddMeasurement("#BinaryReductions", counters.numBinaryReductions);
  AddMeasurement("#GammaReductions", counters.numGammaReductions);

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

size_t
NodeReduction::Statistics::getNumRegions() const noexcept
{
  return NumIterations_.size();
}

size_t
NodeReduction::Statistics::getTotalIterations() const noexcept
{
  size_t sum = 0;
  for (auto [_, numIterations] : NumIterations_)
  {
    sum += numIterations;
  }

  return sum;
}

size_t
NodeReduction::Statistics::getMaxIterationsPerRegion() const noexcept
{
  return std::max_element(
             NumIterations_.begin(),
             NumIterations_.end(),
             [](const auto & p1, const auto & p2)
             {
               return p1.second < p2.second;
             })
      ->second;
}

static std::vector<rvsdg::NodeNormalization<rvsdg::MatchOperation>>
    matchOperationNormalizations({ foldMatchOperationWithConstant });

static std::vector<rvsdg::NodeNormalization<SExtOperation>>
    sextOperationNormalizations({ SExtOperation::foldConstant });

static std::vector<rvsdg::NodeNormalization<ZExtOperation>>
    zextOperationNormalizations({ ZExtOperation::foldConstant });

static std::vector<rvsdg::NodeNormalization<IntegerEqOperation>>
    integerEqNormalizations({ IntegerEqOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerNeOperation>>
    integerNeNormalizations({ IntegerNeOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerSgeOperation>>
    integerSgeNormalizations({ IntegerSgeOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerSgtOperation>>
    integerSgtNormalizations({ IntegerSgtOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerSleOperation>>
    integerSleNormalizations({ IntegerSleOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerSltOperation>>
    integerSltNormalizations({ IntegerSltOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerUgeOperation>>
    integerUgeNormalizations({ IntegerUgeOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerUgtOperation>>
    integerUgtNormalizations({ IntegerUgtOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerUleOperation>>
    integerUleNormalizations({ IntegerUleOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<IntegerUltOperation>>
    integerUltNormalizations({ IntegerUltOperation::foldConstants });

static std::vector<rvsdg::NodeNormalization<LoadNonVolatileOperation>>
    loadNonVolatileNormalizations({ LoadNonVolatileOperation::NormalizeLoadStore,
                                    LoadNonVolatileOperation::NormalizeLoadAlloca,
                                    LoadNonVolatileOperation::NormalizeDuplicateStates,
                                    LoadNonVolatileOperation::NormalizeLoadStoreState,
                                    LoadNonVolatileOperation::NormalizeIOBarrierAllocaAddress });

static std::vector<rvsdg::NodeNormalization<StoreNonVolatileOperation>>
    storeNonVolatileNormalizations({ StoreNonVolatileOperation::NormalizeStoreMux,
                                     StoreNonVolatileOperation::NormalizeStoreStore,
                                     StoreNonVolatileOperation::NormalizeStoreAlloca,
                                     StoreNonVolatileOperation::NormalizeDuplicateStates,
                                     StoreNonVolatileOperation::NormalizeIOBarrierAllocaAddress,
                                     StoreNonVolatileOperation::normalizeStoreAllocaSingleUser });

static std::vector<rvsdg::NodeNormalization<MemoryStateMergeOperation>>
    memoryStateMergeNormalizations({ MemoryStateMergeOperation::NormalizeSingleOperand,
                                     MemoryStateMergeOperation::NormalizeDuplicateOperands,
                                     MemoryStateMergeOperation::NormalizeNestedMerges,
                                     MemoryStateMergeOperation::NormalizeMergeSplit });

static std::vector<rvsdg::NodeNormalization<MemoryStateJoinOperation>>
    memoryStateJoinNormalizations({ MemoryStateJoinOperation::NormalizeSingleOperand,
                                    MemoryStateJoinOperation::NormalizeDuplicateOperands });

static std::vector<rvsdg::NodeNormalization<MemoryStateSplitOperation>>
    memoryStateSplitNormalizations({ MemoryStateSplitOperation::NormalizeSingleResult,
                                     MemoryStateSplitOperation::NormalizeNestedSplits,
                                     MemoryStateSplitOperation::NormalizeSplitMerge });

static std::vector<rvsdg::NodeNormalization<LambdaExitMemoryStateMergeOperation>>
    lambdaExitMemoryStateMergeNormalizations(
        { LambdaExitMemoryStateMergeOperation::NormalizeLoadFromAlloca,
          LambdaExitMemoryStateMergeOperation::NormalizeStoreToAlloca,
          LambdaExitMemoryStateMergeOperation::NormalizeAlloca });

static std::vector<rvsdg::NodeNormalization<PtrCmpOperation>>
    ptrCmpNormalizations({ PtrCmpOperation::normalizeNullPointerComparison });

static std::vector<rvsdg::NodeNormalization<rvsdg::BinaryOperation>>
    binaryOperationNormalizations({ rvsdg::NormalizeBinaryOperation });

template<typename TOperation>
static rvsdg::NodeNormalization<TOperation>
createNormalizer(const std::vector<rvsdg::NodeNormalization<TOperation>> & nodeNormalizations)
{
  return [&](const TOperation & operation, const std::vector<rvsdg::Output *> & operands)
  {
    return rvsdg::NormalizeSequence<TOperation>(nodeNormalizations, operation, operands);
  };
}

template<class TOperation>
static bool
reduceSimpleNode(
    rvsdg::SimpleNode & simpleNode,
    const std::vector<rvsdg::NodeNormalization<TOperation>> & normalizations,
    size_t & counter)
{
  auto normalizer = createNormalizer(normalizations);
  const bool reductionPerformed = rvsdg::ReduceNode<TOperation>(normalizer, simpleNode);
  if (reductionPerformed)
    counter += 1;
  return reductionPerformed;
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
          [this, &reductionPerformed](rvsdg::SimpleNode & simpleNode)
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
  if (const auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(&structuralNode))
  {
    reductionPerformed |= ReduceGammaNode(*gammaNode);
  }

  if (reductionPerformed)
  {
    // We cannot go through the subregions as the structural node might already have been removed.
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
NodeReduction::ReduceGammaNode(rvsdg::GammaNode & gammaNode)
{
  // FIXME: We can not apply the reduction below due to a bug. See github issue #303
  // rvsdg::ReduceGammaControlConstant

  const bool reductionPerformed = reduceStaticallyKnownPredicate(gammaNode);
  if (reductionPerformed)
    Statistics_->getReductionCounters().numGammaReductions++;

  return reductionPerformed;
}

bool
NodeReduction::ReduceSimpleNode(rvsdg::SimpleNode & simpleNode)
{
  if (is<LoadNonVolatileOperation>(&simpleNode))
  {
    return reduceSimpleNode<LoadNonVolatileOperation>(
        simpleNode,
        loadNonVolatileNormalizations,
        Statistics_->getReductionCounters().numLoadNonVolatileReductions);
  }
  if (is<StoreNonVolatileOperation>(&simpleNode))
  {
    return reduceSimpleNode<StoreNonVolatileOperation>(
        simpleNode,
        storeNonVolatileNormalizations,
        Statistics_->getReductionCounters().numStoreNonVolatileReductions);
  }
  if (is<MemoryStateMergeOperation>(&simpleNode))
  {
    return reduceSimpleNode<MemoryStateMergeOperation>(
        simpleNode,
        memoryStateMergeNormalizations,
        Statistics_->getReductionCounters().numMemoryStateMergeReductions);
  }
  if (is<MemoryStateJoinOperation>(&simpleNode))
  {
    return reduceSimpleNode<MemoryStateJoinOperation>(
        simpleNode,
        memoryStateJoinNormalizations,
        Statistics_->getReductionCounters().numMemoryStateJoinReductions);
  }
  if (is<MemoryStateSplitOperation>(&simpleNode))
  {
    return reduceSimpleNode<MemoryStateSplitOperation>(
        simpleNode,
        memoryStateSplitNormalizations,
        Statistics_->getReductionCounters().numMemoryStateSplitReductions);
  }
  if (is<LambdaExitMemoryStateMergeOperation>(&simpleNode))
  {
    return reduceSimpleNode<LambdaExitMemoryStateMergeOperation>(
        simpleNode,
        lambdaExitMemoryStateMergeNormalizations,
        Statistics_->getReductionCounters().numLambdaExitMemoryStateMergeReductions);
  }
  if (is<rvsdg::MatchOperation>(&simpleNode))
  {
    return reduceSimpleNode<rvsdg::MatchOperation>(
        simpleNode,
        matchOperationNormalizations,
        Statistics_->getReductionCounters().numMatchReductions);
  }
  if (is<SExtOperation>(&simpleNode))
  {
    return reduceSimpleNode<SExtOperation>(
        simpleNode,
        sextOperationNormalizations,
        Statistics_->getReductionCounters().numSExtReductions);
  }
  if (is<ZExtOperation>(&simpleNode))
  {
    return reduceSimpleNode<ZExtOperation>(
        simpleNode,
        zextOperationNormalizations,
        Statistics_->getReductionCounters().numZExtReductions);
  }
  if (is<IntegerEqOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerEqOperation>(
        simpleNode,
        integerEqNormalizations,
        Statistics_->getReductionCounters().numIntegerEqReductions);
  }
  if (is<IntegerNeOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerNeOperation>(
        simpleNode,
        integerNeNormalizations,
        Statistics_->getReductionCounters().numIntegerNeReductions);
  }
  if (is<IntegerSgeOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerSgeOperation>(
        simpleNode,
        integerSgeNormalizations,
        Statistics_->getReductionCounters().numIntegerSgeReductions);
  }
  if (is<IntegerSgtOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerSgtOperation>(
        simpleNode,
        integerSgtNormalizations,
        Statistics_->getReductionCounters().numIntegerSgtReductions);
  }
  if (is<IntegerSleOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerSleOperation>(
        simpleNode,
        integerSleNormalizations,
        Statistics_->getReductionCounters().numIntegerSleReductions);
  }
  if (is<IntegerSltOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerSltOperation>(
        simpleNode,
        integerSltNormalizations,
        Statistics_->getReductionCounters().numIntegerSltReductions);
  }
  if (is<IntegerUgeOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerUgeOperation>(
        simpleNode,
        integerUgeNormalizations,
        Statistics_->getReductionCounters().numIntegerUgeReductions);
  }
  if (is<IntegerUgtOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerUgtOperation>(
        simpleNode,
        integerUgtNormalizations,
        Statistics_->getReductionCounters().numIntegerUgtReductions);
  }
  if (is<IntegerUleOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerUleOperation>(
        simpleNode,
        integerUleNormalizations,
        Statistics_->getReductionCounters().numIntegerUleReductions);
  }
  if (is<IntegerUltOperation>(&simpleNode))
  {
    return reduceSimpleNode<IntegerUltOperation>(
        simpleNode,
        integerUltNormalizations,
        Statistics_->getReductionCounters().numIntegerUltReductions);
  }
  if (is<PtrCmpOperation>(&simpleNode))
  {
    return reduceSimpleNode<PtrCmpOperation>(
        simpleNode,
        ptrCmpNormalizations,
        Statistics_->getReductionCounters().numPtrCmpReductions);
  }
  if (is<rvsdg::BinaryOperation>(&simpleNode))
  {
    return reduceSimpleNode<rvsdg::BinaryOperation>(
        simpleNode,
        binaryOperationNormalizations,
        Statistics_->getReductionCounters().numBinaryReductions);
  }

  return false;
}

}
