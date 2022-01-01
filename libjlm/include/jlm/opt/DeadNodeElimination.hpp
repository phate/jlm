/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_DEADNODEELIMINATION_HPP
#define JLM_OPT_DEADNODEELIMINATION_HPP

#include <jlm/opt/optimization.hpp>

namespace jive {
	class region;
}

namespace jlm {

class rvsdg_module;
class StatisticsDescriptor;

/**
* \brief Dead Node Elimination
*/
class dne final : public optimization {
public:
	virtual
	~dne();

	void
	run(jive::region & region);

	virtual void
	run(rvsdg_module & module, const StatisticsDescriptor & sd) override;
};

}

#endif
