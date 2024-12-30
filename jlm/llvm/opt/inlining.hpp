/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_INLINE_HPP
#define JLM_LLVM_OPT_INLINE_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/opt/optimization.hpp>

namespace jlm::llvm
{

class RvsdgModule;

/**
 * \brief Function Inlining
 */
class fctinline final : public optimization
{
public:
  virtual ~fctinline();

  virtual void
  run(RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector) override;
};

jlm::rvsdg::output *
find_producer(jlm::rvsdg::input * input);

void
inlineCall(jlm::rvsdg::SimpleNode * call, const lambda::node * lambda);

}

#endif
