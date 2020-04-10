/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_OPTIMIZATION_HPP
#define JLM_OPT_OPTIMIZATION_HPP

#include <vector>

namespace jlm {

class rvsdg_module;
class stats_descriptor;

/**
* \brief Optimization pass interface
*/
class optimization {
public:
	virtual
	~optimization();

	/**
	* \brief Perform optimization
	*
	* This method is expected to be called multiple times. An
	* implementation is required to reset the objects' internal state
	* to ensure correct behavior after every incovation.
	*
	* \param module RVSDG module the optimization is performed on.
	* \param sd     A stats descriptor for collecting optimization statistics.
	*/
	virtual void
	run(rvsdg_module & module, const stats_descriptor & sd) = 0;
};

/*
	FIXME: This function should be removed.
*/
void
optimize(rvsdg_module & rm,
	const stats_descriptor & sd,
	const std::vector<optimization*> & opts);

}

#endif
