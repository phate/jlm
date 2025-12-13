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
class LambdaNode;
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

  NodeHoisting();

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  hoistNodesInRootRegion(rvsdg::Region & region);

  void
  hoistNodesInLambda(rvsdg::LambdaNode & lambdaNode);

  void
  markNodes(const rvsdg::Region & region);

  void
  hoistNodes(rvsdg::Region & region);

  void
  copyNodeToTargetRegion(rvsdg::Node & node);

  std::vector<rvsdg::Output *>
  getOperandsFromTargetRegion(rvsdg::Node & node) const;

  static rvsdg::Output &
  getOperandFromTargetRegion(rvsdg::Output & output, rvsdg::Region & targetRegion);

  size_t
  computeRegionDepth(const rvsdg::Region & region) const;

  rvsdg::Region &
  computeTargetRegion(const rvsdg::Node & node) const;

  rvsdg::Region &
  computeTargetRegion(const rvsdg::Output & output) const;

  std::unique_ptr<Context> Context_{};
};

#if 0
void
push_top(rvsdg::ThetaNode * theta);

void
push_bottom(rvsdg::ThetaNode * theta);

void
push(rvsdg::ThetaNode * theta);

void
push(rvsdg::GammaNode * gamma);
#endif

}

#endif
