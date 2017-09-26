/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_UNROLL_HPP
#define JLM_OPT_UNROLL_HPP

namespace jive {
	class graph;
}

namespace jlm {

void
unroll(jive::graph & rvsdg, size_t factor);

};

#endif
