/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PULL_HPP
#define JLM_LLVM_OPT_PULL_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/rvsdg/graph.hpp>

namespace jlm::rvsdg
{
class GammaNode;
}

namespace jlm::llvm
{

class RvsdgModule;

/**
 * \brief Node Pull-In Optimization
 */
class pullin final : public optimization
{
public:
  virtual ~pullin();

  virtual void
  run(RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

void
pullin_top(rvsdg::GammaNode * gamma);

void
pullin_bottom(rvsdg::GammaNode * gamma);

void
pull(rvsdg::GammaNode * gamma);

void
pull(jlm::rvsdg::region * region);

}

#endif
