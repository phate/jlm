/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_INLINE_HPP
#define JLM_OPT_INLINE_HPP

namespace jive {
	class graph;
}

namespace jlm {

void
inlining(jive::graph & rvsdg);

}

#endif
