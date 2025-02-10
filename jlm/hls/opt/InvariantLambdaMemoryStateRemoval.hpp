/*
 * Copyright 2025 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_OPT_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP
#define JLM_HLS_OPT_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP

#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

/**
 * @brief Remove invariant memory state edges between Lambda[Entry/Exit]MemoryState nodes.
 *
 * The transformation goes through all lambdas in the RVSDG module and checks if the lambda is only
 * exported. If the lambda is only exported then a check is made for a LambdaExitMemoryStateMerge.
 * If one is found, a check for any invariant edges to its corresponding LambdaEntryMemoryStateSplit
 * node is performed. Any found invariant memory state edges are removed. The memory state split and
 * merge nodes are removed if there is only a single none-invariant edge that remains.
 */
class InvariantLambdaMemoryStateRemoval final : public rvsdg::Transformation
{
  class Statistics;

public:
  virtual ~InvariantLambdaMemoryStateRemoval();

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  /**
   * @brief Remove invariant memory state edges between Lambda[Entry/Exit]MemoryState nodes.
   *
   * The pass checks for a LambdaExitMemoryStateMerge and if found checks if there are any invariant
   * edges to its corresponding LambdaEntryMemoryStateSplit node. Any found invariant memory state
   * edges are removed. The memory state split and merge nodes are removed if there is only a single
   * none-invariant edge.
   *
   * @param memoryState The lambda region result for which invariant memory state edges are to be
   * removed.
   */
  void
  RemoveInvariantMemoryStateEdges(rvsdg::RegionResult * memoryState);

  /**
   * @brief Remove invariant memory state edges between Lambda[Entry/Exit]MemoryState nodes.
   *
   * The pass applies RemoveInvariantMemoryStateEdges(rvsdg::RegionResult * memoryState) to all
   * memory states of all lambdas in the module.
   * @see RemoveInvariantMemoryStateEdges(rvsdg::RegionResult * memoryState)
   *
   * @param rvsdgModule The RVSDG moduled for which invariant memory state edges in all lambda nodes
   * are to be removed.
   */
  void
  RemoveInvariantLambdaMemoryStateEdges(rvsdg::RvsdgModule & rvsdgModule);
};

} // namespace jlm::hls

#endif // JLM_HLS_OPT_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP
