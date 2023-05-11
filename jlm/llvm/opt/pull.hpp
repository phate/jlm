/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PULL_HPP
#define JLM_LLVM_OPT_PULL_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/rvsdg/graph.hpp>

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
    util::StatisticsCollector & statisticsCollector) override;
};

void
pullin_top(jlm::rvsdg::gamma_node * gamma);

void
pullin_bottom(jlm::rvsdg::gamma_node * gamma);


void
pull(jlm::rvsdg::gamma_node * gamma);

void
pull(jlm::rvsdg::region * region);

}

#endif
