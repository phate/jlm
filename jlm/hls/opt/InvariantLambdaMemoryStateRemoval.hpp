/*
 * Copyright 2025 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_OPT_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP
#define JLM_HLS_OPT_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class RegionResult;
}

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
  virtual ~InvariantLambdaMemoryStateRemoval() noexcept;

  /**
   * @brief Applies the transformation on the provided RVSDG module.
   *
   * @param rvsdgModule The RVSDG module to apply the transformation on.
   * @param statisticsCollector The collector used for tracking transformation statistics.
   */
  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  /**
   * @brief Creates an instance of the tranformation and and calls the Run method @see
   * Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
   * override;
   *
   * @param rvsdgModule The RVSDG module to apply the transformation on.
   * @param statisticsCollector The collector used for tracking transformation statistics.
   */
  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector);

private:
  /**
   * @brief Remove invariant memory state edges between Lambda[Entry/Exit]MemoryState nodes.
   *
   * The pass checks for a LambdaExitMemoryStateMerge and if found checks if there are any invariant
   * edges to its corresponding LambdaEntryMemoryStateSplit node. Any found invariant memory state
   * edges are removed. The memory state split and merge nodes are removed if there is only a single
   * none-invariant edge.
   *
   * @param memoryStateResult The lambda region result for which invariant memory state edges are to
   * be removed.
   */
  void
  RemoveInvariantMemoryStateEdges(rvsdg::RegionResult & memoryStateResult);

  /**
   * @brief Remove invariant memory state edges between Lambda[Entry/Exit]MemoryState nodes.
   *
   * The pass applies RemoveInvariantMemoryStateEdges(rvsdg::RegionResult * memoryState) to all
   * memory states of all lambdas in the module that are only exported.
   * @see RemoveInvariantMemoryStateEdges(rvsdg::RegionResult * memoryState)
   *
   * @param rvsdgModule The RVSDG module for which invariant memory state edges are removed in
   * lambdas.
   */
  void
  RemoveInvariantLambdaMemoryStateEdges(rvsdg::RvsdgModule & rvsdgModule);
};

} // namespace jlm::hls

#endif // JLM_HLS_OPT_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP
