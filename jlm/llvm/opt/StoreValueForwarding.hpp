/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_STOREVALUEFORWARDING_HPP
#define JLM_LLVM_OPT_STOREVALUEFORWARDING_HPP

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class Region;
}

namespace jlm::llvm
{

struct StoreValueOrigin;
class LoadTracingInfo;

/** \brief Store Value Forwarding Optimization
 *
 * Store Value Forwarding is an optimization that forwards store values
 * to eliminate redundant loads.
 */
class StoreValueForwarding final : public rvsdg::Transformation
{
  class Statistics;
  struct Context;

public:
  ~StoreValueForwarding() noexcept override;

  StoreValueForwarding();

  StoreValueForwarding(const StoreValueForwarding &) = delete;

  StoreValueForwarding(StoreValueForwarding &&) = delete;

  StoreValueForwarding &
  operator=(const StoreValueForwarding &) = delete;

  StoreValueForwarding &
  operator=(StoreValueForwarding &&) = delete;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  /**
   * Traverse the given inter-procedural region
   *
   * @param region the region to traverse
   */
  void
  traverseInterProceduralRegion(rvsdg::Region & region);

  /**
   * Traverse the given intra-procedural region
   *
   * @param region the region to traverse
   */
  void
  traverseIntraProceduralRegion(rvsdg::Region & region);

  /**
   * Process a non-volatile load node during traversal
   *
   * @param loadNode the load node to handle
   */
  void
  processLoadNode(rvsdg::SimpleNode & loadNode);

  /**
   * Performs store value forwarding to the load node represented by the given \p tracingInfo.
   * Uses the metadata stored during tracing to replace the value output of the load node
   * with the last value that was stored to the memory loaded by it.
   * @param tracingInfo the metadata created during store value origin tracing.
   */
  void
  forwardStoredValues(LoadTracingInfo & tracingInfo);

  /**
   * Gets an output providing the value stored at the given \p storeValueOrigin.
   * The returned output is in the specified \p targetRegion. It must be the same region
   * as the \p storeValueOrigin itself, or a parent region.
   * Getting it may involve routing and creating new structural node inputs and outputs.
   * @param storeValueOrigin the origin of the last stored value along some memory state.
   * @param targetRegion the region the returned output should be in.
   * @param tracingInfo the metadata created during store value origin tracing.
   * @return the rvsdg output providing the stored value in the given region.
   */
  rvsdg::Output &
  getStoredValueOrigin(
      StoreValueOrigin storeValueOrigin,
      rvsdg::Region & targetRegion,
      LoadTracingInfo & tracingInfo);

  /**
   * In \ref getStoredValueOrigin(), all loop variables are created as invariant,
   * to avoid recursive function calls looping around the graph.
   * Instead, the post results of all created loop variables are added to a queue,
   * and properly diverted to their correct origins by this function.
   */
  void
  connectUnroutedLoopPosts(LoadTracingInfo & tracingInfo);

  /**
   * Helper for routing outputs that memoizes the routing to avoid creating
   * duplicate inputs and outputs in structural nodes.
   * The \p output must be in the \p region, or in an ancestor of the region.
   * @param output the output to route
   * @param region the destination of the routing
   * @return an output inside \p region that provides the same value as \p output
   */
  rvsdg::Output &
  routeOutputToRegion(rvsdg::Output & output, rvsdg::Region & region);

  std::unique_ptr<Context> context_;
};

}

#endif
