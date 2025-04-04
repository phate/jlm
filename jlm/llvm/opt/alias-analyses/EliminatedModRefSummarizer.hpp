/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMODREFSUMMARIZER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMODREFSUMMARIZER_HPP

#include <jlm/llvm/opt/alias-analyses/MemoryNodeEliminator.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>

namespace jlm::llvm::aa
{

/** \brief Combines a ModeRefSummarizer and a MemoryNodeEliminator
 *
 * Combines a ModRefSummarizer and a MemoryNodeEliminator by applying them sequentially. The
 * Provider is applied to a given RvsdgModule and PointsToGraph, which results in a
 * ModRefSummary. This ModRefSummary is then fed in to the Eliminator, which
 * removes superfluous memory nodes.
 *
 * @tparam TModRefSummarizer A ModRefSummarizer
 * @tparam Eliminator A MemoryNodeEliminator
 */
template<class TModRefSummarizer, class Eliminator>
class EliminatedModRefSummarizer final : public ModRefSummarizer
{
  static_assert(
      std::is_base_of_v<ModRefSummarizer, TModRefSummarizer>,
      "T is not derived from ModRefSummarizer.");

  static_assert(
      std::is_base_of<MemoryNodeEliminator, Eliminator>::value,
      "T is not derived from MemoryNodeEliminator.");

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
    return Eliminator_.EliminateMemoryNodes(rvsdgModule, *seedModRefSummary, statisticsCollector);
  }

  static std::unique_ptr<ModRefSummary>
  Create(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector)
  {
    EliminatedModRefSummarizer provider{};
    return provider.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
  }

private:
  TModRefSummarizer ModRefSummarizer_;
  Eliminator Eliminator_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMODREFSUMMARIZER_HPP
