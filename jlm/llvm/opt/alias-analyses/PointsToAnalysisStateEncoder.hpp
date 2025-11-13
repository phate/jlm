/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOANALYSISSTATEENCODER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOANALYSISSTATEENCODER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToAnalysis.hpp>
#include <jlm/rvsdg/Transformation.hpp>

#include <type_traits>

namespace jlm::llvm::aa
{

/** Applies points-to analysis and memory state encoding.
 * Uses the information collected during points-to analysis and
 * the memory nodes provided by the mod/ref summarizer
 * to re-encode memory state edges between the operations touching memory.
 *
 * The type of points-to analysis and mod/ref summarizer is specified by the template parameters.
 *
 * @tparam TPointsToAnalysis the subclass of \ref PointsToAnalysis to use
 * @tparam TModRefSummarizer the subclass of \ref ModRefSummarizer to use
 *
 * @see Andersen
 * @see AgnosticModRefSummarizer
 * @see RegionAwareModRefSummarizer
 */
template<typename TPointsToAnalysis, typename TModRefSummarizer>
class PointsToAnalysisStateEncoder final : public rvsdg::Transformation
{
  static_assert(std::is_base_of_v<PointsToAnalysis, TPointsToAnalysis>);
  static_assert(std::is_base_of_v<ModRefSummarizer, TModRefSummarizer>);

public:
  ~PointsToAnalysisStateEncoder() noexcept override;

  PointsToAnalysisStateEncoder()
      : Transformation("PointsToAnalysisStateEncoder")
  {}

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif
