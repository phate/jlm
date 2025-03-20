/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMEMORYNODEPROVIDER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMEMORYNODEPROVIDER_HPP

#include <jlm/llvm/opt/alias-analyses/MemoryNodeEliminator.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryNodeProvider.hpp>

namespace jlm::llvm::aa
{

/** \brief Combines a MemoryNodeProvider and a MemoryNodeEliminator
 *
 * Combines a MemoryNodeProvider and a MemoryNodeEliminator by applying them sequentially. The
 * Provider is applied to a given RvsdgModule and PointsToGraph, which results in a
 * MemoryNodeProvisioning. This MemoryNodeProvisioning is then fed in to the Eliminator, which
 * removes superfluous memory nodes.
 *
 * @tparam Provider A MemoryNodeProvider
 * @tparam Eliminator A MemoryNodeEliminator
 */
template<class Provider, class Eliminator>
class EliminatedMemoryNodeProvider final : public MemoryNodeProvider
{
  static_assert(
      std::is_base_of<MemoryNodeProvider, Provider>::value,
      "T is not derived from MemoryNodeProvider.");

  static_assert(
      std::is_base_of<MemoryNodeEliminator, Eliminator>::value,
      "T is not derived from MemoryNodeEliminator.");

public:
  ~EliminatedMemoryNodeProvider() noexcept override = default;

  EliminatedMemoryNodeProvider() = default;

  EliminatedMemoryNodeProvider(const EliminatedMemoryNodeProvider &) = delete;

  EliminatedMemoryNodeProvider(EliminatedMemoryNodeProvider &&) = delete;

  EliminatedMemoryNodeProvider &
  operator=(const EliminatedMemoryNodeProvider &) = delete;

  EliminatedMemoryNodeProvider &
  operator=(EliminatedMemoryNodeProvider &&) = delete;

  std::unique_ptr<MemoryNodeProvisioning>
  ProvisionMemoryNodes(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector) override
  {
    auto seedProvisioning =
        Provider_.ProvisionMemoryNodes(rvsdgModule, pointsToGraph, statisticsCollector);
    return Eliminator_.EliminateMemoryNodes(rvsdgModule, *seedProvisioning, statisticsCollector);
  }

  static std::unique_ptr<MemoryNodeProvisioning>
  Create(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector)
  {
    EliminatedMemoryNodeProvider provider{};
    return provider.ProvisionMemoryNodes(rvsdgModule, pointsToGraph, statisticsCollector);
  }

private:
  Provider Provider_;
  Eliminator Eliminator_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ELIMINATEDMEMORYNODEPROVIDER_HPP
