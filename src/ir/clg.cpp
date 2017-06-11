/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/cfg.hpp>
#include <jlm/ir/clg.hpp>
#include <jlm/ir/tac.hpp>

#include <jive/common.h>

#include <stdio.h>

#include <algorithm>

/* Tarjan's SCC algorithm */

static void
strongconnect(
	const jlm::clg_node * node,
	std::unordered_map<const jlm::clg_node*, std::pair<size_t,size_t>> & map,
	std::vector<const jlm::clg_node*> & node_stack,
	size_t & index,
	std::vector<std::unordered_set<const jlm::clg_node*>> & sccs)
{
	map.emplace(node, std::make_pair(index, index));
	node_stack.push_back(node);
	index++;

	const std::unordered_set<const jlm::clg_node*> calls = node->calls();
	for (auto callee : calls) {
		if (map.find(callee) == map.end()) {
			/* successor has not been visited yet; recurse on it */
			strongconnect(callee, map, node_stack, index, sccs);
			map[node].second = std::min(map[node].second, map[callee].second);
		} else if (std::find(node_stack.begin(), node_stack.end(), callee) != node_stack.end()) {
			/* successor is in stack and hence in the current SCC */
			map[node].second = std::min(map[node].second, map[callee].first);
		}
	}

	if (map[node].second == map[node].first) {
		std::unordered_set<const jlm::clg_node*> scc;
		const jlm::clg_node * w;
		do {
			w = node_stack.back();
			node_stack.pop_back();
			scc.insert(w);
		} while (w != node);

		sccs.push_back(scc);
	}
}

namespace jlm {

/* clg */

clg_node *
clg::add_function(const char * name, const jive::fct::type & type, bool exported)
{
	std::unique_ptr<clg_node> function(new clg_node(*this, name, type, exported));
	clg_node * f = function.get();
	JLM_DEBUG_ASSERT(nodes_.find(std::string(name)) == nodes_.end());
	nodes_.insert(std::make_pair(std::string(name), std::move(function)));
	return f;
}

clg_node *
clg::lookup_function(const std::string & name) const
{
	if (nodes_.find(name) != nodes_.end())
		return nodes_.find(name)->second.get();

	return nullptr;
}

std::vector<clg_node*>
clg::nodes() const
{
	std::vector<clg_node*> v;
	for (auto i = nodes_.begin(); i != nodes_.end(); i++)
		v.push_back(i->second.get());

	return v;
}

std::vector<std::unordered_set<const clg_node*>>
clg::find_sccs() const
{
	std::vector<std::unordered_set<const clg_node*>> sccs;

	std::unordered_map<const clg_node*, std::pair<size_t,size_t>> map;
	std::vector<const clg_node*> node_stack;
	size_t index = 0;

	std::vector<clg_node*> nodes = this->nodes();
	for (auto node : nodes) {
		if (map.find(node) == map.end())
			strongconnect(node, map, node_stack, index, sccs);
	}

	return sccs;
}

std::string
clg::to_string() const
{
	std::ostringstream osstream;
	for (auto it = nodes_.begin(); it != nodes_.end(); it++) {
		if (it->second->calls().empty())
			osstream << it->first << "\n";
		else {
			for (auto call : it->second->calls())
				osstream << it->first << " -> " << call->name() << "\n";
		}
	}

	return osstream.str();
}

/* function variable */

function_variable::~function_variable()
{}

}
