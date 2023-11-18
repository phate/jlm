/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVISIONING_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVISIONING_HPP

#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/HashSet.hpp>

#include <vector>

namespace jlm::llvm::aa
{

/** \brief Memory Node Provisioning
 *
 * Contains the memory nodes that are required at the entry and exit of a region, and for call
 * nodes.
 */
class MemoryNodeProvisioning
{
public:
  virtual ~MemoryNodeProvisioning() noexcept = default;

  [[nodiscard]] virtual const PointsToGraph &
  GetPointsToGraph() const noexcept = 0;

  [[nodiscard]] virtual const jlm::util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionEntryNodes(const jlm::rvsdg::region & region) const = 0;

  [[nodiscard]] virtual const jlm::util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionExitNodes(const jlm::rvsdg::region & region) const = 0;

  [[nodiscard]] virtual const jlm::util::HashSet<const PointsToGraph::MemoryNode *> &
  GetCallEntryNodes(const CallNode & callNode) const = 0;

  [[nodiscard]] virtual const jlm::util::HashSet<const PointsToGraph::MemoryNode *> &
  GetCallExitNodes(const CallNode & callNode) const = 0;

  [[nodiscard]] virtual jlm::util::HashSet<const PointsToGraph::MemoryNode *>
  GetOutputNodes(const jlm::rvsdg::output & output) const = 0;

  [[nodiscard]] virtual const jlm::util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaEntryNodes(const lambda::node & lambdaNode) const
  {
    return GetRegionEntryNodes(*lambdaNode.subregion());
  }

  [[nodiscard]] virtual const jlm::util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaExitNodes(const lambda::node & lambdaNode) const
  {
    return GetRegionExitNodes(*lambdaNode.subregion());
  }

  [[nodiscard]] virtual const jlm::util::HashSet<const PointsToGraph::MemoryNode *> &
  GetThetaEntryExitNodes(const jlm::rvsdg::theta_node & thetaNode) const
  {
    auto & entryNodes = GetRegionEntryNodes(*thetaNode.subregion());
    auto & exitNodes = GetRegionExitNodes(*thetaNode.subregion());
    JLM_ASSERT(entryNodes == exitNodes);
    return entryNodes;
  }

  [[nodiscard]] virtual jlm::util::HashSet<const PointsToGraph::MemoryNode *>
  GetGammaEntryNodes(const jlm::rvsdg::gamma_node & gammaNode) const
  {
    jlm::util::HashSet<const PointsToGraph::MemoryNode *> allMemoryNodes;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);
      auto & memoryNodes = GetRegionEntryNodes(subregion);
      allMemoryNodes.UnionWith(memoryNodes);
    }

    return allMemoryNodes;
  }

  [[nodiscard]] virtual jlm::util::HashSet<const PointsToGraph::MemoryNode *>
  GetGammaExitNodes(const jlm::rvsdg::gamma_node & gammaNode) const
  {
    jlm::util::HashSet<const PointsToGraph::MemoryNode *> allMemoryNodes;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);
      auto & memoryNodes = GetRegionExitNodes(subregion);
      allMemoryNodes.UnionWith(memoryNodes);
    }

    return allMemoryNodes;
  }
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYNODEPROVISIONING_HPP
