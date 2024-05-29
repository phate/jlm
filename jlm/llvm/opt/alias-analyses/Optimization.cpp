/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/Optimization.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>

namespace jlm::llvm::aa
{

template<typename AliasAnalysisPass, typename MemoryNodeProviderPass>
AliasAnalysisStateEncoder<AliasAnalysisPass, MemoryNodeProviderPass>::
    ~AliasAnalysisStateEncoder() noexcept = default;

template<typename AliasAnalysisPass, typename MemoryNodeProviderPass>
void
AliasAnalysisStateEncoder<AliasAnalysisPass, MemoryNodeProviderPass>::run(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  AliasAnalysisPass aaPass;
  auto pointsToGraph = aaPass.Analyze(rvsdgModule, statisticsCollector);
  auto provisioning =
      MemoryNodeProviderPass::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
}

// Explicitly initialize all combinations
template class AliasAnalysisStateEncoder<Steensgaard, AgnosticMemoryNodeProvider>;
template class AliasAnalysisStateEncoder<Steensgaard, RegionAwareMemoryNodeProvider>;
template class AliasAnalysisStateEncoder<Andersen, AgnosticMemoryNodeProvider>;
template class AliasAnalysisStateEncoder<Andersen, RegionAwareMemoryNodeProvider>;

}
