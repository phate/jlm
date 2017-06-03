/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

#include <algorithm>
#include <deque>
#include <sstream>
#include <unordered_map>

#include <stdio.h>
#include <stdlib.h>
#include <sstream>

/* Tarjan's SCC algorithm */

static void
strongconnect(
	jlm::cfg_node * node,
	jlm::cfg_node * exit,
	std::unordered_map<jlm::cfg_node*, std::pair<size_t,size_t>> & map,
	std::vector<jlm::cfg_node*> & node_stack,
	size_t & index,
	std::vector<std::unordered_set<jlm::cfg_node*>> & sccs)
{
	map.emplace(node, std::make_pair(index, index));
	node_stack.push_back(node);
	index++;

	if (node != exit) {
		std::vector<jlm::cfg_edge*> edges = node->outedges();
		for (size_t n = 0; n < edges.size(); n++) {
			jlm::cfg_node * successor = edges[n]->sink();
			if (map.find(successor) == map.end()) {
				/* successor has not been visited yet; recurse on it */
				strongconnect(successor, exit, map, node_stack, index, sccs);
				map[node].second = std::min(map[node].second, map[successor].second);
			} else if (std::find(node_stack.begin(), node_stack.end(), successor) != node_stack.end()) {
				/* successor is in stack and hence in the current SCC */
				map[node].second = std::min(map[node].second, map[successor].first);
			}
		}
	}

	if (map[node].second == map[node].first) {
		std::unordered_set<jlm::cfg_node*> scc;
		jlm::cfg_node * w;
		do {
			w = node_stack.back();
			node_stack.pop_back();
			scc.insert(w);
		} while (w != node);

		if (scc.size() != 1 || (*scc.begin())->has_selfloop_edge())
			sccs.push_back(scc);
	}
}

static inline std::unique_ptr<jlm::cfg>
copy_structural(const jlm::cfg & in)
{
	std::unique_ptr<jlm::cfg> out(new jlm::cfg(in.module()));
	out->entry_node()->remove_outedge(out->entry_node()->outedge(0));

	/* create all nodes */
	std::unordered_map<const jlm::cfg_node*, jlm::cfg_node*> node_map;
	for (const auto & node : in) {
		if (&node == in.entry_node()) {
			node_map[&node] = out->entry_node();
		} else if (&node == in.exit_node()) {
			node_map[&node] = out->exit_node();
		} else {
			node_map[&node] = out->create_node(node.attribute());
		}
	}

	/* establish control flow */
	for (const auto & node : in) {
		for (const auto & e : node.outedges()) {
			node_map[&node]->add_outedge(node_map[e->sink()], e->index());
		}
	}

	return out;
}

static inline bool
is_loop(const jlm::cfg_node * node) noexcept
{
	return node->ninedges() == 2
	    && node->noutedges() == 2
	    && node->has_selfloop_edge();
}

static inline bool
is_linear(const jlm::cfg_node * node) noexcept
{
	if (node->noutedges() != 1)
		return false;

	if (node->outedge(0)->sink()->ninedges() != 1)
		return false;

	return true;
}

static inline jlm::cfg_node *
find_join(const jlm::cfg_node * split) noexcept
{
	JLM_DEBUG_ASSERT(split->noutedges() > 1);
	auto s1 = split->outedge(0)->sink();
	auto s2 = split->outedge(1)->sink();

	jlm::cfg_node * join = nullptr;
	if (s1->noutedges() == 1 && s1->outedge(0)->sink() == s2)
		join = s2;
	else if (s2->noutedges() == 1 && s2->outedge(0)->sink() == s1)
		join = s1;
	else if (s1->noutedges() == 1 && s2->noutedges() == 1
	     && (s1->outedge(0)->sink() == s2->outedge(0)->sink()))
		join = s1->outedge(0)->sink();

	return join;
}

static inline bool
is_branch(const jlm::cfg_node * split) noexcept
{
	if (split->noutedges() < 2)
		return false;

	auto join = find_join(split);
	if (join == nullptr || join->ninedges() != split->noutedges())
		return false;

	for (const auto & e : split->outedges()) {
		if (e->sink() == join)
			continue;

		auto node = e->sink();
		if (node->ninedges() != 1)
			return false;
		if (node->noutedges() != 1 || node->outedge(0)->sink() != join)
			return false;
	}

	return true;
}

static inline bool
is_T1(const jlm::cfg_node * node) noexcept
{
	for (const auto & e : node->outedges()) {
		if (e->source() == e->sink())
			return true;
	}

	return false;
}

static inline bool
is_T2(const jlm::cfg_node * node) noexcept
{
	if (node->ninedges() == 0)
		return false;

	auto source = node->inedges().front()->source();
	for (const auto & e : node->inedges()) {
		if (e->source() != source)
			return false;
	}

	return true;
}

static inline void
reduce_loop(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_loop(node));
	auto cfg = node->cfg();

	auto reduction = create_basic_block_node(cfg);
	for (auto & e : node->outedges()) {
		if (e->is_selfloop()) {
			node->remove_outedge(e);
			break;
		}
	}

	reduction->add_outedge(node->outedge(0)->sink(), 0);
	node->remove_outedges();
	node->divert_inedges(reduction);

	to_visit.erase(node);
	to_visit.insert(reduction);
}

static inline void
reduce_linear(
	jlm::cfg_node * entry,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_linear(entry));
	auto exit = entry->outedge(0)->sink();
	auto cfg = entry->cfg();

	auto reduction = create_basic_block_node(cfg);
	entry->divert_inedges(reduction);
	for (const auto & e : exit->outedges())
		reduction->add_outedge(e->sink(), e->index());
	exit->remove_outedges();

	to_visit.erase(entry);
	to_visit.erase(exit);
	to_visit.insert(reduction);
}

static inline void
reduce_branch(
	jlm::cfg_node * split,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_branch(split));
	auto join = find_join(split);
	auto cfg = split->cfg();

	auto reduction = create_basic_block_node(cfg);
	split->divert_inedges(reduction);
	reduction->add_outedge(join, 0);
	for (const auto & e : split->outedges()) {
		if (e->sink() != join) {
			e->sink()->remove_outedges();
			to_visit.erase(e->sink());
		}
	}

	to_visit.erase(split);
	to_visit.insert(reduction);
}

static inline void
reduce_T1(jlm::cfg_node * node)
{
	JLM_DEBUG_ASSERT(is_T1(node));

	for (auto & e : node->outedges()) {
		if (e->source() == e->sink()) {
			node->remove_outedge(e);
			break;
		}
	}
}

static inline void
reduce_T2(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_T2(node));

	auto p = node->inedges().front()->source();
	p->divert_inedges(node);
	p->remove_outedges();
	to_visit.erase(p);
}

static inline bool
reduce_structured(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	if (is_loop(node)) {
		reduce_loop(node, to_visit);
		return true;
	}

	if (is_branch(node)) {
		reduce_branch(node, to_visit);
		return true;
	}

	if (is_linear(node)) {
		reduce_linear(node, to_visit);
		return true;
	}

	return false;
}

static inline bool
reduce_reducible(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	if (is_T1(node)) {
		reduce_T1(node);
		return true;
	}

	if (is_T2(node)) {
		reduce_T2(node, to_visit);
		return true;
	}

	return false;
}

namespace jlm {

/* entry attribute */

static inline jlm::cfg_node *
create_entry_node(jlm::cfg * cfg)
{
	jlm::entry_attribute attr;
	return cfg->create_node(attr);
}

entry_attribute::~entry_attribute()
{}

std::string
entry_attribute::debug_string() const noexcept
{
	return "ENTRY";
}

std::unique_ptr<attribute>
entry_attribute::copy() const
{
	return std::unique_ptr<attribute>(new entry_attribute(*this));
}

/* exit attribute */

static inline jlm::cfg_node *
create_exit_node(jlm::cfg * cfg)
{
	jlm::exit_attribute attr;
	return cfg->create_node(attr);
}

exit_attribute::~exit_attribute()
{}

std::string
exit_attribute::debug_string() const noexcept
{
	return "EXIT";
}

std::unique_ptr<attribute>
exit_attribute::copy() const
{
	return std::unique_ptr<attribute>(new exit_attribute(*this));
}

/* cfg */

cfg::cfg(jlm::module & module)
: module_(module)
{
	entry_ = create_entry_node(this);
	exit_ = create_exit_node(this);
	entry_->add_outedge(exit_, 0);
}

cfg_node *
cfg::create_node(const attribute & attr)
{
	auto node = std::make_unique<cfg_node>(*this, attr);
	auto tmp = node.get();
	nodes_.insert(std::move(node));
	return tmp;
}

bool
cfg::is_structured() const
{
	JLM_DEBUG_ASSERT(is_closed(*this));
	auto c = copy_structural(*this);

	std::unordered_set<cfg_node*> to_visit;
	for (auto & node : *c)
		to_visit.insert(&node);

	auto it = to_visit.begin();
	while (it != to_visit.end()) {
		bool reduced = reduce_structured(*it, to_visit);
		it = reduced ? to_visit.begin() : std::next(it);
	}

	return to_visit.size() == 1;
}

bool
cfg::is_reducible() const
{
	JLM_DEBUG_ASSERT(is_closed(*this));
	auto c = copy_structural(*this);

	std::unordered_set<cfg_node*> to_visit;
	for (auto & node : *c)
		to_visit.insert(&node);

	auto it = to_visit.begin();
	while (it != to_visit.end()) {
		bool reduced = reduce_reducible(*it, to_visit);
		it = reduced ? to_visit.begin() : std::next(it);
	}

	return to_visit.size() == 1;
}

std::string
cfg::convert_to_dot() const
{
	std::string dot("digraph cfg{\n");
	for (const auto & node : nodes_) {
		dot += strfmt((intptr_t)node.get());
		dot += strfmt("[shape = box, label = \"", node->debug_string(), "\"];\n");
		for (const auto & e : node->outedges()) {
			dot += strfmt((intptr_t)e->source(), " -> ", (intptr_t)e->sink());
			dot += strfmt("[label = \"", e->index(), "\"];\n");
		}
	}
	dot += "}\n";

	return dot;
}

void
cfg::remove_node(cfg_node * node)
{
	node->remove_inedges();
	node->remove_outedges();

	std::unique_ptr<cfg_node> tmp(node);
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it = nodes_.find(tmp);
	JLM_DEBUG_ASSERT(it != nodes_.end());
	nodes_.erase(it);
	tmp.release();
}

void
cfg::prune()
{
	JLM_DEBUG_ASSERT(is_valid(*this));

	/* find all nodes that are dominated by the entry node */
	std::unordered_set<cfg_node*> to_visit({entry_node()});
	std::unordered_set<cfg_node*> visited;
	while (!to_visit.empty()) {
		cfg_node * node = *to_visit.begin();
		to_visit.erase(to_visit.begin());
		JLM_DEBUG_ASSERT(visited.find(node) == visited.end());
		visited.insert(node);
		std::vector<cfg_edge*> edges = node->outedges();
		for (auto edge : edges) {
			if (visited.find(edge->sink()) == visited.end()
			&& to_visit.find(edge->sink()) == to_visit.end())
				to_visit.insert(edge->sink());
		}
	}

	/* remove all nodes not dominated by the entry node */
	std::unordered_set<std::unique_ptr<cfg_node>>::iterator it = nodes_.begin();
	while (it != nodes_.end()) {
		if (visited.find((*it).get()) == visited.end()) {
			cfg_node * node = (*it).get();
			node->remove_inedges();
			node->remove_outedges();
			it = nodes_.erase(it);
		} else
			it++;
	}

	JLM_DEBUG_ASSERT(is_closed(*this));
}

bool
is_valid(const jlm::cfg & cfg)
{
	for (const auto & node : cfg) {
		if (&node == cfg.exit_node()) {
			if (!node.no_successor())
				return false;
			continue;
		}

		if (&node == cfg.entry_node()) {
			if (!node.no_predecessor())
				return false;
			if (!node.single_successor())
				return false;
			if (node.outedges()[0]->index() != 0)
				return false;
			continue;
		}

		if (node.no_successor())
			return false;

		/*
			Check whether all indices are 0 and in ascending order (uniqueness of indices)
		*/
		std::vector<cfg_edge*> edges = node.outedges();
		std::sort(edges.begin(), edges.end(),
			[](const cfg_edge * e1, const cfg_edge * e2) { return e1->index() < e2->index(); });
		for (size_t n = 0; n < edges.size(); n++) {
			if (edges[n]->index() != n)
				return false;
		}

		/*
			Check whether the CFG is actually a graph and not a multigraph
		*/
		for (size_t n = 1; n < edges.size(); n++) {
			if (edges[n-1]->sink() == edges[n]->sink())
				return false;
		}
	}

	return true;
}

bool
is_closed(const jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_valid(cfg));

	for (const auto & node : cfg) {
		if (&node == cfg.entry_node())
			continue;

		if (node.no_predecessor())
			return false;
	}

	return true;
}

bool
is_linear(const jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_closed(cfg));

	for (const auto & node : cfg) {
		if (&node == cfg.entry_node() || &node == cfg.exit_node())
			continue;

		if (!node.single_successor() || !node.single_predecessor())
			return false;
	}

	return true;
}

std::vector<std::unordered_set<cfg_node*>>
find_sccs(const jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_closed(cfg));

	size_t index = 0;
	std::vector<cfg_node*> node_stack;
	std::vector<std::unordered_set<cfg_node*>> sccs;
	std::unordered_map<cfg_node*, std::pair<size_t,size_t>> map;
	strongconnect(cfg.entry_node(), cfg.exit_node(), map, node_stack, index, sccs);

	return sccs;
}

}

void
jive_cfg_view(const jlm::cfg & cfg)
{
	FILE * file = popen("tee /tmp/cfg.dot | dot -Tps > /tmp/cfg.ps ; gv /tmp/cfg.ps", "w");
	auto dot = cfg.convert_to_dot();
	fwrite(dot.c_str(), dot.size(), 1, file);
	pclose(file);
}
