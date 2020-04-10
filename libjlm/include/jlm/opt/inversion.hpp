/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_INVERSION_HPP
#define JLM_OPT_INVERSION_HPP

#include <jlm/opt/optimization.hpp>

namespace jlm {

class rvsdg_module;
class stats_descriptor;

/**
* \brief Theta-Gamma Inversion
*/
class tginversion final : public optimization {
public:
	virtual
	~tginversion();

	virtual void
	run(rvsdg_module & module, const stats_descriptor & sd) override;
};

}

#endif
