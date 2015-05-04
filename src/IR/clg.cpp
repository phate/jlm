/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/cfg.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/tac.hpp>

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
clg::add_function(const char * name, jive::fct::type & type)
{
	std::unique_ptr<clg_node> function(new clg_node(*this, name, type));
	clg_node * f = function.get();
	nodes_.insert(std::make_pair(std::string(name), std::move(function)));
	function.release();
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

/* clg node */

std::vector<const variable*>
clg_node::cfg_begin(const std::vector<std::string> & names)
{
	if (names.size() != type_->narguments())
		throw jive::compiler_error("Invalid number of argument names.");

	std::vector<const variable*> arguments;

	cfg_.reset(new jlm::cfg(*this));
	for (size_t n = 0; n < names.size(); n++)
		arguments.push_back(cfg_->append_argument(names[n], *type_->argument_type(n)));

	return arguments;
}

void
clg_node::cfg_end(const std::vector<const variable*> & results)
{
	JLM_DEBUG_ASSERT(cfg_.get() != nullptr);

	if (results.size() != type_->nreturns())
		throw jive::compiler_error("Invalid number of results.");

	for (size_t n = 0; n < results.size(); n++) {
		if (results[n]->type() != *type_->return_type(n))
			throw jive::type_error(type_->return_type(n)->debug_string(),
				results[n]->type().debug_string());
		cfg_->append_result(results[n]);
	}
}

}
