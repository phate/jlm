/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP

#include <jlm/llvm/opt/alias-analyses/MemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToAnalysis.hpp>
#include <jlm/rvsdg/Transformation.hpp>

#include <type_traits>

namespace jlm::llvm::aa
{

/** Applies points-to analysis and memory state encoding.
 * Uses the information collected during points-to analysis and
 * the memory nodes provided by the memory node provider
 * to reencode memory state edges between the operations touching memory.
 *
 * The type of points-to analysis and memory node provider is specified by the template parameters.
 *
 * @tparam TPointsToAnalysis the subclass of PointsToAnalysis to use
 * @tparam MemoryNodeProviderPass the subclass of MemoryNodeProvider to use
 *
 * @see Steensgaard
 * @see Andersen
 * @see AgnosticMemoryNodeProvider
 * @see RegionAwareMemoryNodeProvider
 */
template<typename TPointsToAnalysis, typename MemoryNodeProviderPass>
class PointsToAnalysisStateEncoder final : public rvsdg::Transformation
{
  static_assert(std::is_base_of_v<PointsToAnalysis, TPointsToAnalysis>);
  static_assert(std::is_base_of_v<MemoryNodeProvider, MemoryNodeProviderPass>);

public:
  ~PointsToAnalysisStateEncoder() noexcept override;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif
