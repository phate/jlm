/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_LOOPUNSWITCHING_HPP
#define JLM_LLVM_OPT_LOOPUNSWITCHING_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class Region;
class ThetaNode;
}

namespace jlm::llvm
{

/**
 * \brief LoopUnswitching
 */
class LoopUnswitching final : public rvsdg::Transformation
{
public:
  class Statistics;

  ~LoopUnswitching() noexcept override;

  LoopUnswitching()
      : Transformation("LoopUnswitching")
  {}

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector);

private:
  static void
  HandleRegion(rvsdg::Region & region);

  static bool
  UnswitchLoop(rvsdg::ThetaNode & thetaNode);

  static rvsdg::GammaNode *
  IsUnswitchable(const rvsdg::ThetaNode & thetaNode);

  static void
  SinkNodesIntoGamma(rvsdg::GammaNode & gammaNode, const rvsdg::ThetaNode & thetaNode);
};

}

#endif
