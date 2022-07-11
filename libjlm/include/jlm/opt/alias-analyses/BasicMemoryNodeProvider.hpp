/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_BASICMEMORYNODEPROVIDER_HPP
#define JLM_OPT_ALIAS_ANALYSES_BASICMEMORYNODEPROVIDER_HPP

#include <jlm/opt/alias-analyses/MemoryNodeProvider.hpp>

namespace jlm::aa
{

class BasicMemoryNodeProvider final : public MemoryNodeProvider {
public:
  explicit
  BasicMemoryNodeProvider(const PointsToGraph & pointsToGraph);

  BasicMemoryNodeProvider(const BasicMemoryNodeProvider&) = delete;

  BasicMemoryNodeProvider(BasicMemoryNodeProvider&&) = delete;

  BasicMemoryNodeProvider &
  operator=(const BasicMemoryNodeProvider&) = delete;

  BasicMemoryNodeProvider &
  operator=(BasicMemoryNodeProvider&&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override;

  [[nodiscard]] const std::vector<const PointsToGraph::MemoryNode*> &
  GetRegionEntryNodes(const jive::region & region) const override;

  [[nodiscard]] const std::vector<const PointsToGraph::MemoryNode*> &
  GetRegionExitNodes(const jive::region & region) const override;

  [[nodiscard]] const std::vector<const PointsToGraph::MemoryNode*> &
  GetCallEntryNodes(const CallNode & callNode) const override;

  [[nodiscard]] const std::vector<const PointsToGraph::MemoryNode*> &
  GetCallExitNodes(const CallNode & callNode) const override;

  [[nodiscard]] std::vector<const PointsToGraph::MemoryNode*>
  GetOutputNodes(const jive::output & output) const override;

private:
  void
  CollectMemoryNodes(const PointsToGraph & pointsToGraph);

  const PointsToGraph & PointsToGraph_;

  std::vector<const PointsToGraph::MemoryNode*> MemoryNodes_;
};

}

#endif //JLM_OPT_ALIAS_ANALYSES_BASICMEMORYNODEPROVIDER_HPP
