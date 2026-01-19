/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_STOREVALUEFORWARDING_HPP
#define JLM_LLVM_OPT_STOREVALUEFORWARDING_HPP

#include "jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp"
#include "jlm/rvsdg/simple-node.hpp"
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class Region;
}

namespace jlm::llvm
{

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
   * Process a node during top-down traversal
   *
   * @param node the node to process
   */
  void
  processSimpleNode(rvsdg::SimpleNode & node);

  /**
   * Process a non-volatile load node during traversal
   *
   * @param loadNode the load node to handle
   */
  void
  processLoadNode(rvsdg::SimpleNode & loadNode);

  /**
   * Traces from the the given state output until a store node is reached.
   * Load operations are traced through, but calls and non-invariant structural nodes are not.
   * @return the memory state output of the store, or nullptr if no store node was reached.
   */
  rvsdg::Output *
  traceStateEdgeToStoreNode(rvsdg::Output & state);

  /**
   * Performs a simple alias analysis query between a load and store node,
   * to determine if they read and write from the same exact address,
   * possibly interfere, or if the operations are guaranteed to be fully independent.
   * @param loadedAddress the address that is being loaded
   * @param loadedTypeSize the number of bytes that are loaded from the address
   * @param storeNode the StoreOperation node
   */
  aa::AliasAnalysis::AliasQueryResponse
  queryAliasAnalysis(
      rvsdg::Output & loadedAddress,
      size_t loadedTypeSize,
      rvsdg::SimpleNode & storeNode);

  /**
   * Diverts all users of the \ref loadNode to instead take the value stored by the \ref storeNode.
   * The store may be in a parent region of the load, in which case the value is routed in.
   *
   * @pre the store and load nodes have matching value types, and the store preceeds the load
   * @param storeNode the StoreOperation node
   * @param loadNode the LoadOperation node to be replaced
   */
  void
  performStoreLoadForwarding(rvsdg::SimpleNode & storeNode, rvsdg::SimpleNode & loadNode);

  std::unique_ptr<Context> context_;
};

}

#endif
