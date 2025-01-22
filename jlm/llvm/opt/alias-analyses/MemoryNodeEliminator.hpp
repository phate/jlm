/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEELIMINATOR_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEELIMINATOR_HPP

#include <memory>

namespace jlm::rvsdg
{
class RvsdgModule;
}

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm::aa
{

class MemoryNodeProvisioning;

class MemoryNodeEliminator
{
public:
  virtual ~MemoryNodeEliminator() noexcept = default;

  /**
   * Eliminates unnecessary memory nodes from a MemoryNodeProvisioning.
   *
   * @param rvsdgModule The RVSDG module from which the seedProvisioning was computed from.
   * @param seedProvisioning A provisioning from which memory nodes will be eliminated.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return An instance of MemoryNodeProvisioning.
   */
  virtual std::unique_ptr<MemoryNodeProvisioning>
  EliminateMemoryNodes(
      const rvsdg::RvsdgModule & rvsdgModule,
      const MemoryNodeProvisioning & seedProvisioning,
      jlm::util::StatisticsCollector & statisticsCollector) = 0;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEELIMINATOR_HPP
