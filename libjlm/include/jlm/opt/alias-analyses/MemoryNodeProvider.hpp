/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP
#define JLM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP

#include <jlm/opt/alias-analyses/PointsToGraph.hpp>

#include <vector>

namespace jlm::aa {

class MemoryNodeProvider {
public:
  virtual
  ~MemoryNodeProvider() noexcept;

  [[nodiscard]] virtual const PointsToGraph &
  GetPointsToGraph() const noexcept = 0;

  [[nodiscard]] virtual const std::vector<const PointsToGraph::MemoryNode*> &
  GetRegionEntryNodes(const jive::region & region) const = 0;

  [[nodiscard]] virtual const std::vector<const PointsToGraph::MemoryNode*> &
  GetRegionExitNodes(const jive::region & region) const = 0;

  [[nodiscard]] virtual const std::vector<const PointsToGraph::MemoryNode*> &
  GetCallEntryNodes(const CallNode & callNode) const = 0;

  [[nodiscard]] virtual const std::vector<const PointsToGraph::MemoryNode*> &
  GetCallExitNodes(const CallNode & callNode) const = 0;

  [[nodiscard]] virtual std::vector<const PointsToGraph::MemoryNode*>
  GetOutputNodes(const jive::output & output) const = 0;

  [[nodiscard]] virtual const std::vector<const PointsToGraph::MemoryNode*> &
  GetLambdaEntryNodes(const lambda::node & lambdaNode) const
  {
    return GetRegionEntryNodes(*lambdaNode.subregion());
  }

  [[nodiscard]] virtual const std::vector<const PointsToGraph::MemoryNode*> &
  GetLambdaExitNodes(const lambda::node & lambdaNode) const
  {
    return GetRegionExitNodes(*lambdaNode.subregion());
  }

  [[nodiscard]] virtual const std::vector<const PointsToGraph::MemoryNode*> &
  GetThetaEntryExitNodes(const jive::theta_node & thetaNode) const
  {
    auto & entryNodes = GetRegionEntryNodes(*thetaNode.subregion());
    auto & exitNodes = GetRegionExitNodes(*thetaNode.subregion());
    JLM_ASSERT(entryNodes == exitNodes);
    return entryNodes;
  }

  [[nodiscard]] virtual std::vector<const PointsToGraph::MemoryNode*>
  GetGammaEntryNodes(const jive::gamma_node & gammaNode) const
  {
    std::unordered_set<const PointsToGraph::MemoryNode*> allMemoryNodes;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++) {
      auto & subregion = *gammaNode.subregion(n);
      auto & memoryNodes = GetRegionEntryNodes(subregion);
      allMemoryNodes.insert(memoryNodes.begin(), memoryNodes.end());
    }

    return {allMemoryNodes.begin(), allMemoryNodes.end()};
  }

  [[nodiscard]] virtual std::vector<const PointsToGraph::MemoryNode*>
  GetGammaExitNodes(const jive::gamma_node & gammaNode) const
  {
    std::unordered_set<const PointsToGraph::MemoryNode*> allMemoryNodes;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++) {
      auto & subregion = *gammaNode.subregion(n);
      auto & memoryNodes = GetRegionExitNodes(subregion);
      allMemoryNodes.insert(memoryNodes.begin(), memoryNodes.end());
    }

    return {allMemoryNodes.begin(), allMemoryNodes.end()};
  }
};

}

#endif //JLM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVIDER_HPP
