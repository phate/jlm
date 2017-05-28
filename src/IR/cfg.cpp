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
#include <jive/util/buffer.h>

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

cfg::cfg()
{
	entry_ = create_entry_node(this);
	exit_ = create_exit_node(this);
	entry_->add_outedge(exit_, 0);
}

cfg::cfg(const cfg & other)
{
	/* FIXME: function does not take care of tacs and variables */

	/* create all nodes */
	std::unordered_map<const cfg_node*, cfg_node*> node_map;
	for (const auto & node : other) {
		if (&node == other.entry_node()) {
			entry_  = create_entry_node(this);
			node_map[&node] = entry_;
		} else if (&node == other.exit_node()) {
			exit_ = create_exit_node(this);
			node_map[&node] = exit_;
		} else {
			node_map[&node] = create_node(node.attribute());
		}
	}

	/* establish control flow */
	for (const auto & node : other) {
		for (const auto & e : node.outedges()) {
			node_map[&node]->add_outedge(node_map[e->sink()], e->index());
		}
	}
}

cfg_node *
cfg::create_node(const attribute & attr)
{
	auto node = std::make_unique<cfg_node>(*this, attr);
	auto tmp = node.get();
	nodes_.insert(std::move(node));
	return tmp;
}

std::vector<std::unordered_set<cfg_node*>>
cfg::find_sccs() const
{
	JLM_DEBUG_ASSERT(is_closed(*this));

	std::vector<std::unordered_set<cfg_node*>> sccs;

	std::unordered_map<cfg_node*, std::pair<size_t,size_t>> map;
	std::vector<cfg_node*> node_stack;
	size_t index = 0;

	strongconnect(entry_node(), exit_node(), map, node_stack, index, sccs);

	return sccs;
}

bool
cfg::is_acyclic() const
{
	std::vector<std::unordered_set<cfg_node*>> sccs = find_sccs();
	return sccs.size() == 0;
}

bool
cfg::is_structured() const
{
	JLM_DEBUG_ASSERT(is_closed(*this));

	cfg c(*this);
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it = c.nodes_.begin();
	while (it != c.nodes_.end()) {
		cfg_node * node = (*it).get();

		if (c.nodes_.size() == 2) {
			JLM_DEBUG_ASSERT(is_closed(c));
			return true;
		}

		if (node == c.entry_node() || node == c.exit_node()) {
			it++; continue;
		}

		/* loop */
		bool is_selfloop = false;
		std::vector<cfg_edge*> edges = node->outedges();
		for (auto edge : edges) {
			if (edge->is_selfloop())  {
				node->remove_outedge(edge);
				is_selfloop = true; break;
			}
		}
		if (is_selfloop) {
			it = c.nodes_.begin(); continue;
		}

		/* linear */
		if (node->single_successor() && node->outedges()[0]->sink()->single_predecessor()) {
			JLM_DEBUG_ASSERT(node->noutedges() == 1 && node->outedges()[0]->sink()->ninedges() == 1);
			node->divert_inedges(node->outedges()[0]->sink());
			c.remove_node(node);
			it = c.nodes_.begin(); continue;
		}

		/* branch */
		if (node->is_branch()) {
			/* find tail node */
			JLM_DEBUG_ASSERT(node->noutedges() > 1);
			std::vector<cfg_edge*> edges = node->outedges();
			cfg_node * succ1 = edges[0]->sink();
			cfg_node * succ2 = edges[1]->sink();
			cfg_node * tail = nullptr;
			if (succ1->noutedges() == 1 && succ1->outedges()[0]->sink() == succ2)
				tail = succ2;
			else if (succ2->noutedges() == 1 && succ2->outedges()[0]->sink() == succ1)
				tail = succ1;
			else if (succ1->noutedges() == 1 && succ2->noutedges() == 1
				&& succ1->outedges()[0]->sink() == succ2->outedges()[0]->sink())
				tail = succ1->outedges()[0]->sink();

			if (tail == nullptr || tail->ninedges() != node->noutedges()) {
				it++; continue;
			}

			/* check whether it corresponds to a branch subgraph */
			bool is_branch = true;
			for (auto edge : edges) {
				cfg_node * succ = edge->sink();
				if (succ != tail
				&& ((succ->ninedges() != 1 || succ->inedges().front()->source() != node)
					|| (succ->noutedges() != 1 || succ->outedges().front()->sink() != tail))) {
					is_branch = false; break;
				}
			}
			if (!is_branch) {
				it++; continue;
			}

			/* remove branch subgraph */
			for (auto edge : edges) {
				if (edge->sink() != tail)
					c.remove_node(edge->sink());
			}
			node->remove_outedges();
			node->add_outedge(tail, 0);
			it = c.nodes_.begin(); continue;
		}

		it++;
	}

	return false;
}

bool
cfg::is_reducible() const
{
	JLM_DEBUG_ASSERT(is_closed(*this));

	cfg c(*this);
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it = c.nodes_.begin();
	while (it != c.nodes_.end()) {
		cfg_node * node = (*it).get();

		if (c.nodes_.size() == 2) {
			JLM_DEBUG_ASSERT(is_closed(c));
			return true;
		}

		if (node == c.entry_node() || node == c.exit_node()) {
			it++; continue;
		}

		/* T1 */
		bool is_selfloop = false;
		std::vector<cfg_edge*> edges = node->outedges();
		for (auto edge : edges) {
			if (edge->is_selfloop()) {
				node->remove_outedge(edge);
				is_selfloop = true; break;
			}
		}
		if (is_selfloop) {
			it = c.nodes_.begin(); continue;
		}

		/* T2 */
		if (node->single_predecessor()) {
			cfg_node * predecessor = node->inedges().front()->source();
			std::vector<cfg_edge*> edges = node->outedges();
			for (size_t e = 0; e < edges.size(); e++) {
				predecessor->add_outedge(edges[e]->sink(), 0);
			}
			c.remove_node(node);
			it = c.nodes_.begin(); continue;
		}

		it++;
	}

	return false;
}

void
cfg::convert_to_dot(jive::buffer & buffer) const
{
	buffer.append("digraph cfg {\n");

	char tmp[96];
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it;
	for (it = nodes_.begin(); it != nodes_.end(); it++) {
		cfg_node * node = (*it).get();
		snprintf(tmp, sizeof(tmp), "%zu", (size_t)node);
		buffer.append(tmp).append("[shape = box, label = \"");
		buffer.append(node->debug_string().c_str()).append("\"];\n");

		std::vector<cfg_edge*> edges = node->outedges();
		for (size_t n = 0; n < edges.size(); n++) {
			snprintf(tmp, sizeof(tmp), "%zu -> %zu[label = \"%zu\"];\n", (size_t)edges[n]->source(),
				(size_t)edges[n]->sink(), edges[n]->index());
			buffer.append(tmp);
		}
	}

	buffer.append("}\n");
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

}

void
jive_cfg_view(const jlm::cfg & self)
{
	jive::buffer buffer;
	FILE * file = popen("tee /tmp/cfg.dot | dot -Tps > /tmp/cfg.ps ; gv /tmp/cfg.ps", "w");
	self.convert_to_dot(buffer);
	fwrite(buffer.c_str(), buffer.size(), 1, file);
	pclose(file);
}
