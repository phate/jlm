/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP

#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/HashSet.hpp>

#include <vector>

namespace jlm {
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

  [[nodiscard]] virtual const HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionEntryNodes(const jive::region & region) const = 0;

  [[nodiscard]] virtual const HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionExitNodes(const jive::region & region) const = 0;

  [[nodiscard]] virtual const HashSet<const PointsToGraph::MemoryNode*> &
  GetCallEntryNodes(const CallNode & callNode) const = 0;

  [[nodiscard]] virtual const HashSet<const PointsToGraph::MemoryNode*> &
  GetCallExitNodes(const CallNode & callNode) const = 0;

  [[nodiscard]] virtual HashSet<const PointsToGraph::MemoryNode*>
  GetOutputNodes(const jive::output & output) const = 0;

  [[nodiscard]] virtual const HashSet<const PointsToGraph::MemoryNode*> &
  GetLambdaEntryNodes(const lambda::node & lambdaNode) const
  {
    return GetRegionEntryNodes(*lambdaNode.subregion());
  }

  [[nodiscard]] virtual const HashSet<const PointsToGraph::MemoryNode*> &
  GetLambdaExitNodes(const lambda::node & lambdaNode) const
  {
    return GetRegionExitNodes(*lambdaNode.subregion());
  }

  [[nodiscard]] virtual const HashSet<const PointsToGraph::MemoryNode*> &
  GetThetaEntryExitNodes(const jive::theta_node & thetaNode) const
  {
    auto & entryNodes = GetRegionEntryNodes(*thetaNode.subregion());
    auto & exitNodes = GetRegionExitNodes(*thetaNode.subregion());
    JLM_ASSERT(entryNodes == exitNodes);
    return entryNodes;
  }

  [[nodiscard]] virtual HashSet<const PointsToGraph::MemoryNode*>
  GetGammaEntryNodes(const jive::gamma_node & gammaNode) const
  {
    HashSet<const PointsToGraph::MemoryNode*> allMemoryNodes;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++) {
      auto & subregion = *gammaNode.subregion(n);
      auto & memoryNodes = GetRegionEntryNodes(subregion);
      allMemoryNodes.UnionWith(memoryNodes);
    }

    return allMemoryNodes;
  }

  [[nodiscard]] virtual HashSet<const PointsToGraph::MemoryNode*>
  GetGammaExitNodes(const jive::gamma_node & gammaNode) const
  {
    HashSet<const PointsToGraph::MemoryNode*> allMemoryNodes;
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
    StatisticsCollector & statisticsCollector) = 0;
};

}

#endif //JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP
