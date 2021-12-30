/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_CNE_HPP
#define JLM_OPT_CNE_HPP

#include <jlm/opt/optimization.hpp>

namespace jlm {

class rvsdg_module;
class StatisticsDescriptor;

/**
* \brief Common Node Elimination
*/
class cne final : public optimization {
public:
	virtual
	~cne();

	virtual void
	run(rvsdg_module & module, const StatisticsDescriptor & sd) override;
};

}

#endif
