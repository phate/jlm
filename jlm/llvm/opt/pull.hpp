/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PULL_HPP
#define JLM_LLVM_OPT_PULL_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class Region;
}

namespace jlm::llvm
{

/**
 * \brief Node Pull-In Optimization
 */
class pullin final : public rvsdg::Transformation
{
public:
  virtual ~pullin();

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

void
pullin_top(rvsdg::GammaNode * gamma);

void
pullin_bottom(rvsdg::GammaNode * gamma);

void
pull(rvsdg::GammaNode * gamma);

void
pull(rvsdg::Region * region);

}

#endif
