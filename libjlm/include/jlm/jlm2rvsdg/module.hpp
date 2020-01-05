/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM2RVSDG_MODULE_H
#define JLM_JLM2RVSDG_MODULE_H

#include <memory>

namespace jive {
	class graph;
}

namespace jlm {

class module;
class rvsdg_module;
class stats_descriptor;

std::unique_ptr<rvsdg_module>
construct_rvsdg(const module & m, const stats_descriptor & sd);

}

#endif
