/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PUSH_HPP
#define JLM_LLVM_OPT_PUSH_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jive {

class gamma_node;
class theta_node;

}

namespace jlm {

class RvsdgModule;

/**
* \brief Node Push-Out Optimization
*/
class pushout final : public optimization {
public:
	virtual
	~pushout();

	virtual void
	run(
    RvsdgModule & module,
    StatisticsCollector & statisticsCollector) override;
};

void
push_top(jive::theta_node * theta);

void
push_bottom(jive::theta_node * theta);

void
push(jive::theta_node * theta);

void
push(jive::gamma_node * gamma);

}

#endif
