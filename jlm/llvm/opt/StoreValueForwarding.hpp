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

  std::unique_ptr<Context> context_;
};

}

#endif
