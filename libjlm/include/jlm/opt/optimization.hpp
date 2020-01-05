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

enum class optimization {cne, dne, iln, inv, psh, red, ivt, url, pll};

void
optimize(rvsdg_module & rm, const optimization & opt);

void
optimize(
	rvsdg_module & rm,
	const std::vector<optimization> & opts,
	const stats_descriptor & sd);

}

#endif
