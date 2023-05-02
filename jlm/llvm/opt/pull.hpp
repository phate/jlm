/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PULL_HPP
#define JLM_LLVM_OPT_PULL_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/rvsdg/graph.hpp>

namespace jive {
	class gamma_node;
	class region;
}

namespace jlm {

class RvsdgModule;

/**
* \brief Node Pull-In Optimization
*/
class pullin final : public optimization {
public:
	virtual
	~pullin();

	virtual void
	run(
    RvsdgModule & module,
    StatisticsCollector & statisticsCollector) override;
};

void
pullin_top(jive::gamma_node * gamma);

void
pullin_bottom(jive::gamma_node * gamma);


void
pull(jive::gamma_node * gamma);

void
pull(jive::region * region);

}

#endif
