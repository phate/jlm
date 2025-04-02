/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm::aa
{

class MemoryNodeProvider
{
public:
  virtual ~MemoryNodeProvider() noexcept = default;

  /**
   * Computes the memory nodes that are required at the entry and exit of a region,
   * or at the entry/exit of a call node.
   *
   * @param rvsdgModule The RVSDG module on which the memory node provision should be performed.
   * @param pointsToGraph The points-to graph corresponding to \p rvsdgModule.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return An instance of ModRefSummary.
   */
  virtual std::unique_ptr<ModRefSummary>
  ProvisionMemoryNodes(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      jlm::util::StatisticsCollector & statisticsCollector) = 0;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP
