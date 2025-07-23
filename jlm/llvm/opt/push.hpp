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
public:
  class Statistics;

  ~NodeHoisting() noexcept override;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
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
