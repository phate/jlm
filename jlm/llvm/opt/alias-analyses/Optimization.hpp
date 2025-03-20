/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryNodeProvider.hpp>
#include <jlm/rvsdg/Transformation.hpp>

#include <type_traits>

namespace jlm::llvm::aa
{

/** Applies alias analysis and memory state encoding.
 * Uses the information collected during alias analysis and
 * the memory nodes provided by the memory node provider
 * to reencode memory state edges between the operations touching memory.
 *
 * The type of alias analysis and memory node provider is specified by the template parameters.
 *
 * @tparam AliasAnalysisPass the subclass of AliasAnalysis to use
 * @tparam MemoryNodeProviderPass the subclass of MemoryNodeProvider to use
 *
 * @see Steensgaard
 * @see Andersen
 * @see AgnosticMemoryNodeProvider
 * @see RegionAwareMemoryNodeProvider
 */
template<typename AliasAnalysisPass, typename MemoryNodeProviderPass>
class AliasAnalysisStateEncoder final : public rvsdg::Transformation
{
  static_assert(std::is_base_of_v<AliasAnalysis, AliasAnalysisPass>);
  static_assert(std::is_base_of_v<MemoryNodeProvider, MemoryNodeProviderPass>);

public:
  ~AliasAnalysisStateEncoder() noexcept override;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif
