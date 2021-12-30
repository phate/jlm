/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_PULL_HPP
#define JLM_OPT_PULL_HPP

#include <jive/rvsdg/graph.hpp>

#include <jlm/opt/optimization.hpp>

namespace jive {
	class gamma_node;
	class region;
}

namespace jlm {

class rvsdg_module;
class StatisticsDescriptor;

/**
* \brief Node Pull-In Optimization
*/
class pullin final : public optimization {
public:
	virtual
	~pullin();

	virtual void
	run(rvsdg_module & module, const StatisticsDescriptor & sd) override;
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
