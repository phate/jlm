/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_REDUCTION_HPP
#define JLM_OPT_REDUCTION_HPP

#include <jlm/opt/optimization.hpp>

namespace jlm {

class rvsdg;
class StatisticsDescriptor;

/**
* \brief Node Reduction Optimization
*/
class nodereduction final : public optimization {
public:
	virtual
	~nodereduction();

	virtual void
	run(RvsdgModule & module, const StatisticsDescriptor & sd) override;
};

}

#endif
