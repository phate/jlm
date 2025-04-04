/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMEMORYNODEPROVIDER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMEMORYNODEPROVIDER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefEliminator.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>

namespace jlm::llvm::aa
{

/** \brief Combines a MemoryNodeProvider and a ModRefEliminator
 *
 * Combines a MemoryNodeProvider and a ModRefEliminator by applying them sequentially. The
 * Provider is applied to a given RvsdgModule and PointsToGraph, which results in a
 * MemoryNodeProvisioning. This MemoryNodeProvisioning is then fed in to the Eliminator, which
 * removes superfluous memory nodes.
 *
 * @tparam Provider A MemoryNodeProvider
 * @tparam TModRefEliminator A ModRefEliminator
 */
template<class Provider, class TModRefEliminator>
class EliminatedMemoryNodeProvider final : public ModRefSummarizer
{
  static_assert(
      std::is_base_of_v<ModRefSummarizer, Provider>,
      "T is not derived from ModRefSummarizer.");

  static_assert(
      std::is_base_of_v<ModRefEliminator, TModRefEliminator>,
      "T is not derived from ModRefEliminator.");

public:
  ~EliminatedMemoryNodeProvider() noexcept override = default;

  EliminatedMemoryNodeProvider() = default;

  EliminatedMemoryNodeProvider(const EliminatedMemoryNodeProvider &) = delete;

  EliminatedMemoryNodeProvider(EliminatedMemoryNodeProvider &&) = delete;

  EliminatedMemoryNodeProvider &
  operator=(const EliminatedMemoryNodeProvider &) = delete;

  EliminatedMemoryNodeProvider &
  operator=(EliminatedMemoryNodeProvider &&) = delete;

  std::unique_ptr<ModRefSummary>
  SummarizeModRefs(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector) override
  {
    auto seedModRefSummary =
        Provider_.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
    return ModRefEliminator_.EliminateModRefs(rvsdgModule, *seedModRefSummary, statisticsCollector);
  }

  static std::unique_ptr<ModRefSummary>
  Create(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector)
  {
    EliminatedMemoryNodeProvider provider{};
    return provider.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
  }

private:
  Provider Provider_;
  TModRefEliminator ModRefEliminator_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMEMORYNODEPROVIDER_HPP
