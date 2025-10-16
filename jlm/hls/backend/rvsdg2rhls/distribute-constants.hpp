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
  static void
  distributeConstantsInRootRegion(rvsdg::Region & region);

  static void
  distributeConstantsInLambda(rvsdg::LambdaNode & lambdaNode);

  static util::HashSet<rvsdg::SimpleNode *>
  collectConstants(rvsdg::Region & region);

  static util::HashSet<rvsdg::Output *>
  collectOutputsWithSimpleNodeUsers(const rvsdg::SimpleNode & simpleNode);
};

}

#endif
