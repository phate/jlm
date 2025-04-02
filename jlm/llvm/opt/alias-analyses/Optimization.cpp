/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/EliminatedMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/Optimization.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/llvm/opt/alias-analyses/TopDownMemoryNodeEliminator.hpp>

namespace jlm::llvm::aa
{

template<typename TPointsToAnalysis, typename MemoryNodeProviderPass>
PointsToAnalysisStateEncoder<TPointsToAnalysis, MemoryNodeProviderPass>::
    ~PointsToAnalysisStateEncoder() noexcept = default;

template<typename TPointsToAnalysisPass, typename MemoryNodeProviderPass>
void
PointsToAnalysisStateEncoder<TPointsToAnalysisPass, MemoryNodeProviderPass>::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  TPointsToAnalysisPass ptaPass;
  auto pointsToGraph = ptaPass.Analyze(rvsdgModule, statisticsCollector);
  auto provisioning =
      MemoryNodeProviderPass::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
}

// Explicitly initialize all combinations
template class PointsToAnalysisStateEncoder<Steensgaard, AgnosticMemoryNodeProvider>;
template class PointsToAnalysisStateEncoder<Steensgaard, RegionAwareMemoryNodeProvider>;
template class PointsToAnalysisStateEncoder<Andersen, AgnosticMemoryNodeProvider>;
template class PointsToAnalysisStateEncoder<Andersen, RegionAwareMemoryNodeProvider>;
template class PointsToAnalysisStateEncoder<
    Andersen,
    EliminatedMemoryNodeProvider<AgnosticMemoryNodeProvider, TopDownMemoryNodeEliminator>>;

}
