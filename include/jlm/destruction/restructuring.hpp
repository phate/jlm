/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_DESTRUCTION_RESTRUCTURING_HPP
#define JLM_DESTRUCTION_RESTRUCTURING_HPP

#include <unordered_set>

namespace jlm {

namespace frontend {
	class cfg;
	class cfg_edge;
}

std::unordered_set<const jlm::frontend::cfg_edge*>
restructure(jlm::frontend::cfg * cfg);

}

#endif
