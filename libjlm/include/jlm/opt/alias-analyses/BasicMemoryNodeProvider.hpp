/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_BASICMEMORYNODEPROVIDER_HPP
#define JLM_OPT_ALIAS_ANALYSES_BASICMEMORYNODEPROVIDER_HPP

#include <jlm/opt/alias-analyses/MemoryNodeProvider.hpp>

namespace jlm::aa
{

/** \brief Basic memory node provider
 *
 * The key idea of the basic memory node provider is that \b all memory states are routed through \b all structural
 * nodes irregardless of whether these states are required by any simple nodes within the structural nodes. This
 * strategy ensures that the state of a memory location is always present for encoding while avoiding the complexity of
 * an additional analysis for determining the required routing path of the states. The drawback is that
 * a lot of states are routed through structural nodes where they are not needed, potentially leading to a significant
 * runtime of the encoder for bigger RVSDGs.
 *
 * @see MemoryNodeProvider
 * @see MemoryStateEncoder
 */
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
