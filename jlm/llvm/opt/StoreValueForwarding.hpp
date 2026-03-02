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

struct LoadTracingInfo;
struct StoreValueOrigin;

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
   * Getting this output may involve routing and creating new structural node inputs and outputs.
   * @param storeValueOrigin the origin of the last stored value along some memory state.
   * @param tracingInfo the metdata created during store value origin tracing.
   * @return the rvsdg output providing the stored value
   */
  rvsdg::Output &
  getStoredValueOrigin(StoreValueOrigin storeValueOrigin, LoadTracingInfo & tracingInfo);

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
