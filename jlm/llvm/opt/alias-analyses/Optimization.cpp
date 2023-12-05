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

template<typename AliasAnalysisPass, bool regionAware>
MemoryStateEncodingPass<AliasAnalysisPass, regionAware>::~MemoryStateEncodingPass() noexcept =
    default;

template<typename AliasAnalysisPass, bool regionAware>
void
MemoryStateEncodingPass<AliasAnalysisPass, regionAware>::run(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  AliasAnalysisPass aaPass;
  auto pointsToGraph = aaPass.Analyze(rvsdgModule, statisticsCollector);

  using ProvisioningPass =
      std::conditional_t<regionAware, RegionAwareMemoryNodeProvider, AgnosticMemoryNodeProvider>;

  auto provisioning = ProvisioningPass::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
}

// Explicitly initialize all possible combinations
template class MemoryStateEncodingPass<Steensgaard, false>;
template class MemoryStateEncodingPass<Steensgaard, true>;
template class MemoryStateEncodingPass<Andersen, false>;
template class MemoryStateEncodingPass<Andersen, true>;

}
