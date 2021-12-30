/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_INLINE_HPP
#define JLM_OPT_INLINE_HPP

#include <jlm/opt/optimization.hpp>

namespace jlm {

class rvsdg_module;
class StatisticsDescriptor;

/**
* \brief Function Inlining
*/
class fctinline final : public optimization {
public:
	virtual
	~fctinline();

	virtual void
	run(rvsdg_module & module, const StatisticsDescriptor & sd) override;
};

}

#endif
