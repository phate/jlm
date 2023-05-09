/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP

#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/HashSet.hpp>

#include <vector>

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::aa {

/** \brief Memory Node Provisioning
 *
 * Contains the memory nodes that are required at the entry and exit of a region, and for call nodes.
 */
class MemoryNodeProvisioning {
public:
  virtual
  ~MemoryNodeProvisioning() noexcept;

  [[nodiscard]] virtual const PointsToGraph &
  GetPointsToGraph() const noexcept = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionEntryNodes(const jlm::rvsdg::region & region) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionExitNodes(const jlm::rvsdg::region & region) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode*> &
  GetCallEntryNodes(const CallNode & callNode) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode*> &
  GetCallExitNodes(const CallNode & callNode) const = 0;

  [[nodiscard]] virtual util::HashSet<const PointsToGraph::MemoryNode*>
  GetOutputNodes(const jlm::rvsdg::output & output) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode*> &
  GetLambdaEntryNodes(const lambda::node & lambdaNode) const
  {
    return GetRegionEntryNodes(*lambdaNode.subregion());
  }

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode*> &
  GetLambdaExitNodes(const lambda::node & lambdaNode) const
  {
    return GetRegionExitNodes(*lambdaNode.subregion());
  }

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode*> &
  GetThetaEntryExitNodes(const jlm::rvsdg::theta_node & thetaNode) const
  {
    auto & entryNodes = GetRegionEntryNodes(*thetaNode.subregion());
    auto & exitNodes = GetRegionExitNodes(*thetaNode.subregion());
    JLM_ASSERT(entryNodes == exitNodes);
    return entryNodes;
  }

  [[nodiscard]] virtual util::HashSet<const PointsToGraph::MemoryNode*>
  GetGammaEntryNodes(const jlm::rvsdg::gamma_node & gammaNode) const
  {
    util::HashSet<const PointsToGraph::MemoryNode*> allMemoryNodes;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++) {
      auto & subregion = *gammaNode.subregion(n);
      auto & memoryNodes = GetRegionEntryNodes(subregion);
      allMemoryNodes.UnionWith(memoryNodes);
    }

    return allMemoryNodes;
  }

  [[nodiscard]] virtual util::HashSet<const PointsToGraph::MemoryNode*>
  GetGammaExitNodes(const jlm::rvsdg::gamma_node & gammaNode) const
  {
    util::HashSet<const PointsToGraph::MemoryNode*> allMemoryNodes;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++) {
      auto & subregion = *gammaNode.subregion(n);
      auto & memoryNodes = GetRegionExitNodes(subregion);
      allMemoryNodes.UnionWith(memoryNodes);
    }

    return allMemoryNodes;
  }
};

class MemoryNodeProvider {
public:
  virtual
  ~MemoryNodeProvider() noexcept;

  /**
   * Computes the memory nodes that are required at the entry and exit of of a region as well as call node.
   *
   * @param rvsdgModule The RVSDG module on which the memory node provision should be performed.
   * @param pointsToGraph The points-to graph corresponding to \p rvsdgModule.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return An instance of MemoryNodeProvisioning.
   */
  virtual std::unique_ptr<MemoryNodeProvisioning>
  ProvisionMemoryNodes(
    const RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector) = 0;
};

}

#endif //JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP
