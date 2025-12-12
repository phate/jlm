/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PUSH_HPP
#define JLM_LLVM_OPT_PUSH_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class ThetaNode;
}

namespace jlm::llvm
{

/**
 * \brief Node Hoisting Optimization
 */
class NodeHoisting final : public rvsdg::Transformation
{
  class Context;

public:
  class Statistics;

  ~NodeHoisting() noexcept override;

  NodeHoisting()
      : Transformation("NodeHoisting")
  {}

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  hoistNodesInRootRegion(rvsdg::Region & region);

  void
  hoistNodesInLambda(rvsdg::LambdaNode & lambdaNode);

  void
  markNodesInRegion(const rvsdg::Region & region);

  static bool
  isEligibleToHoist(const rvsdg::Node & node);

  void
  computeRegionDepth(const rvsdg::Region & region);

  void
  computeTargetRegion(const rvsdg::Node & node);

  rvsdg::Region &
  computeTargetRegion(const rvsdg::Output & output) const;

  std::unique_ptr<Context> Context_;
};

void
push_top(rvsdg::ThetaNode * theta);

void
push_bottom(rvsdg::ThetaNode * theta);

void
push(rvsdg::ThetaNode * theta);

void
push(rvsdg::GammaNode * gamma);

}

#endif
