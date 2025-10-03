/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysisPrecisionEvaluator.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/LocalAliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToAnalysisStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraphAliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>

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

  if (statisticsCollector.IsDemanded(util::Statistics::Id::AliasAnalysisPrecisionEvaluation))
  {
    AliasAnalysisPrecisionEvaluator precisionEvaluator;

    // Use different alias analyses, and their combination
    LocalAliasAnalysis localAA;
    PointsToGraphAliasAnalysis ptgAA(*pointsToGraph);
    ChainedAliasAnalysis ptgPlusLocalAA(ptgAA, localAA);

    precisionEvaluator.EvaluateAliasAnalysisClient(rvsdgModule, localAA, statisticsCollector);
    precisionEvaluator.EvaluateAliasAnalysisClient(rvsdgModule, ptgAA, statisticsCollector);
    precisionEvaluator.EvaluateAliasAnalysisClient(
        rvsdgModule,
        ptgPlusLocalAA,
        statisticsCollector);
  }

  auto modRefSummary = TModRefSummarizer::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *modRefSummary, statisticsCollector);
}

// Explicitly initialize all combinations
template class PointsToAnalysisStateEncoder<Steensgaard, AgnosticModRefSummarizer>;
template class PointsToAnalysisStateEncoder<Steensgaard, RegionAwareModRefSummarizer>;
template class PointsToAnalysisStateEncoder<Andersen, AgnosticModRefSummarizer>;
template class PointsToAnalysisStateEncoder<Andersen, RegionAwareModRefSummarizer>;

}
