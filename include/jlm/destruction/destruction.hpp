/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_DESTRUCTION_DESTRUCTION_H
#define JLM_DESTRUCTION_DESTRUCTION_H

namespace jive {
	class graph;
}

namespace jlm {

class module;

jive::graph *
construct_rvsdg(const module & m);

}

#endif
