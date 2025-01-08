/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PUSH_HPP
#define JLM_LLVM_OPT_PUSH_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class ThetaNode;
}

namespace jlm::llvm
{

class RvsdgModule;

/**
 * \brief Node Push-Out Optimization
 */
class pushout final : public optimization
{
public:
  virtual ~pushout();

  virtual void
  run(RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
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
