/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_INLINE_HPP
#define JLM_LLVM_OPT_INLINE_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{

/**
 * \brief Function Inlining
 */
class FunctionInlining final : public rvsdg::Transformation
{
public:
  class Statistics;
  
  ~FunctionInlining() noexcept override;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

jlm::rvsdg::Output *
find_producer(jlm::rvsdg::Input * input);

void
inlineCall(jlm::rvsdg::SimpleNode * call, const rvsdg::LambdaNode * lambda);

}

#endif
