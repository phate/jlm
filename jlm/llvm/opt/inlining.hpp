/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_INLINE_HPP
#define JLM_LLVM_OPT_INLINE_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>

namespace jlm {

class RvsdgModule;

/**
* \brief Function Inlining
*/
class fctinline final : public optimization {
public:
	virtual
	~fctinline();

	virtual void
	run(
    RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector) override;
};

jlm::rvsdg::output *
find_producer(jlm::rvsdg::input * input);

void
inlineCall(jlm::rvsdg::simple_node * call, const lambda::node * lambda);

}

#endif
