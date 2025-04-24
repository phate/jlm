/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMODREFSUMMARIZER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMODREFSUMMARIZER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefEliminator.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>

namespace jlm::llvm::aa
{

/** \brief Combines a ModeRefSummarizer and a ModRefEliminator
 *
 * Combines a ModRefSummarizer and a ModRefEliminator by applying them sequentially. The
 * Provider is applied to a given RvsdgModule and PointsToGraph, which results in a
 * ModRefSummary. This ModRefSummary is then fed in to the Eliminator, which
 * removes superfluous memory nodes.
 *
 * @tparam TModRefSummarizer A ModRefSummarizer
 * @tparam TModRefEliminator A ModRefEliminator
 */
template<class TModRefSummarizer, class TModRefEliminator>
class EliminatedModRefSummarizer final : public ModRefSummarizer
{
  static_assert(
      std::is_base_of_v<ModRefSummarizer, TModRefSummarizer>,
      "T is not derived from ModRefSummarizer.");

  static_assert(
      std::is_base_of_v<ModRefEliminator, TModRefEliminator>,
      "T is not derived from ModRefEliminator.");

public:
  ~EliminatedModRefSummarizer() noexcept override = default;

  EliminatedModRefSummarizer() = default;

  EliminatedModRefSummarizer(const EliminatedModRefSummarizer &) = delete;

  EliminatedModRefSummarizer(EliminatedModRefSummarizer &&) = delete;

  EliminatedModRefSummarizer &
  operator=(const EliminatedModRefSummarizer &) = delete;

  EliminatedModRefSummarizer &
  operator=(EliminatedModRefSummarizer &&) = delete;

  std::unique_ptr<ModRefSummary>
  SummarizeModRefs(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector) override
  {
    auto seedModRefSummary =
        ModRefSummarizer_.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
    return ModRefEliminator_.EliminateModRefs(rvsdgModule, *seedModRefSummary, statisticsCollector);
  }

  static std::unique_ptr<ModRefSummary>
  Create(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector)
  {
    EliminatedModRefSummarizer summarizer{};
    return summarizer.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
  }

private:
  TModRefSummarizer ModRefSummarizer_;
  TModRefEliminator ModRefEliminator_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMODREFSUMMARIZER_HPP
