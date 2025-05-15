/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/EliminatedModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/Optimization.hpp>
#include <jlm/llvm/opt/alias-analyses/PrecisionEvaluator.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/llvm/opt/alias-analyses/TopDownModRefEliminator.hpp>

namespace jlm::llvm::aa
{

template<typename TPointsToAnalysis, typename TModRefSummarizer>
PointsToAnalysisStateEncoder<TPointsToAnalysis, TModRefSummarizer>::
    ~PointsToAnalysisStateEncoder() noexcept = default;

template<typename TPointsToAnalysis, typename TModRefSummarizer>
void
PointsToAnalysisStateEncoder<TPointsToAnalysis, TModRefSummarizer>::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  TPointsToAnalysis ptaPass;
  auto pointsToGraph = ptaPass.Analyze(rvsdgModule, statisticsCollector);

  // Evaluate alias analysis precision if the statistic is demanded
  PrecisionEvaluator precisionEvaluator;
  PointsToGraphAliasAnalysis ptgAA(*pointsToGraph);
  BasicAliasAnalysis basicAA;
  LlvmAliasAnalysis llvmAA;
  ChainedAliasAnalysis ptgPlusLlvmAA(ptgAA, llvmAA);

  precisionEvaluator.EvaluateAliasAnalysisClient(rvsdgModule, basicAA, statisticsCollector);
  precisionEvaluator.EvaluateAliasAnalysisClient(rvsdgModule, llvmAA, statisticsCollector);
  precisionEvaluator.EvaluateAliasAnalysisClient(rvsdgModule, ptgAA, statisticsCollector);
  precisionEvaluator.EvaluateAliasAnalysisClient(rvsdgModule, ptgPlusLlvmAA, statisticsCollector);

  /*
  TODO: Add encoding back in
  auto modRefSummary = TModRefSummarizer::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *modRefSummary, statisticsCollector);
  */
}

// Explicitly initialize all combinations
template class PointsToAnalysisStateEncoder<Steensgaard, AgnosticModRefSummarizer>;
template class PointsToAnalysisStateEncoder<Steensgaard, RegionAwareModRefSummarizer>;
template class PointsToAnalysisStateEncoder<Andersen, AgnosticModRefSummarizer>;
template class PointsToAnalysisStateEncoder<Andersen, RegionAwareModRefSummarizer>;
template class PointsToAnalysisStateEncoder<
    Andersen,
    EliminatedModRefSummarizer<AgnosticModRefSummarizer, TopDownModRefEliminator>>;

}
