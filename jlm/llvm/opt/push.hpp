/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PUSH_HPP
#define JLM_LLVM_OPT_PUSH_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jlm::llvm
{

namespace rvsdg
{
class gamma_node;
class theta_node;
}

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
push_top(jlm::rvsdg::theta_node * theta);

void
push_bottom(jlm::rvsdg::theta_node * theta);

void
push(jlm::rvsdg::theta_node * theta);

void
push(jlm::rvsdg::gamma_node * gamma);

}

#endif
