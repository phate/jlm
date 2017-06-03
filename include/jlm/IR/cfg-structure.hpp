/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_CFG_STRUCTURE_HPP
#define JLM_IR_CFG_STRUCTURE_HPP

#include <unordered_set>
#include <vector>

namespace jlm {

class cfg;
class cfg_node;

bool
is_valid(const jlm::cfg & cfg);

bool
is_closed(const jlm::cfg & cfg);

std::vector<std::unordered_set<jlm::cfg_node*>>
find_sccs(const jlm::cfg & cfg);

static inline bool
is_acyclic(const jlm::cfg & cfg)
{
	auto sccs = find_sccs(cfg);
	return sccs.size() == 0;
}

bool
is_structured(const jlm::cfg & cfg);

bool
is_reducible(const jlm::cfg & cfg);

}

#endif
