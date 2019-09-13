/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_OPTIMIZATION_HPP
#define JLM_OPT_OPTIMIZATION_HPP

#include <vector>

namespace jlm {

class rvsdg;
class stats_descriptor;

enum class optimization {cne, dne, iln, inv, psh, red, ivt, url, pll};

void
optimize(jlm::rvsdg & rvsdg, const optimization & opt);

void
optimize(
	jlm::rvsdg & rvsdg,
	const std::vector<optimization> & opts,
	const stats_descriptor & sd);

}

#endif
