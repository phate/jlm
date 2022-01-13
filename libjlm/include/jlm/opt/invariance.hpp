/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_INVARIANCE_HPP
#define JLM_OPT_INVARIANCE_HPP

#include <jlm/opt/optimization.hpp>

namespace jlm {

class RvsdgModule;
class StatisticsDescriptor;

/**
* \brief Invariant Value Redirection
*/
class ivr final : public optimization {
public:
	virtual
	~ivr();

	virtual void
	run(RvsdgModule & module, const StatisticsDescriptor & sd) override;
};

}

#endif
