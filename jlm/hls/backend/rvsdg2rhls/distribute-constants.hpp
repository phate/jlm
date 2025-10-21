/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_DISTRIBUTE_CONSTANTS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_DISTRIBUTE_CONSTANTS_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class LambdaNode;
}

namespace jlm::hls
{

/**
 * Distributes constants into subregions of gamma and theta nodes if they have users there.
 */
class ConstantDistribution final : public rvsdg::Transformation
{
public:
  ~ConstantDistribution() noexcept override;

  ConstantDistribution();

  ConstantDistribution(const ConstantDistribution &) = delete;

  ConstantDistribution &
  operator=(const ConstantDistribution &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    ConstantDistribution constantDistribution;
    constantDistribution.Run(rvsdgModule, statisticsCollector);
  }

private:
  /**
   * Traverse the root region as well as phi regions to find all lambda nodes where constants should
   * be distributed.
   *
   * @param region The region to traverse.
   */
  static void
  distributeConstantsInRootRegion(rvsdg::Region & region);

  /**
   * Distribute all constants within a lambda node.
   *
   * @param lambdaNode The lambda node where the constants are distributed.
   */
  static void
  distributeConstantsInLambda(const rvsdg::LambdaNode & lambdaNode);

  /**
   * Collects all constants within a region and recursively within its structural node's subregion.
   *
   * @param region the region for which to collect constants
   * @return A set of constants.
   */
  static util::HashSet<rvsdg::SimpleNode *>
  collectConstants(rvsdg::Region & region);

  /**
   * For a given constant node \p constantNode, collect all gamma or theta node arguments/outputs
   * where the constant node has users that are either another simple node or a lambda subregion
   * result. The idea is that only outputs are of interest where there is a "real user" of a
   * constant, and it is not just routed through a region.
   *
   * @param constantNode The constant node for which to collect the outputs.
   * @return A set outputs that have "real users" of \p constantNode.
   */
  static util::HashSet<rvsdg::Output *>
  collectOutputs(const rvsdg::SimpleNode & constantNode);
};

}

#endif
