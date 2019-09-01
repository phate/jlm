/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_OPTIMIZATION_HPP
#define JLM_OPT_OPTIMIZATION_HPP

#include <vector>

namespace jive {
	class graph;
}

namespace jlm {

enum class optimization {cne, dne, iln, inv, psh, red, ivt, url, pll};

void
optimize(jive::graph & graph, const optimization & opt);

static inline void
optimize(jive::graph & graph, const std::vector<optimization> & opts)
{
	for (const auto & opt : opts)
		optimize(graph, opt);
}

}

#endif
