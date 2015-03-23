/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/IR/assignment.hpp>
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

/* enter node */

cfg::enter_node::~enter_node() noexcept {}

cfg::enter_node::enter_node(jlm::cfg & cfg) noexcept
	: cfg_node(cfg)
{}

std::string
cfg::enter_node::debug_string() const
{
	std::stringstream sstrm;

	sstrm << this << " (ENTER)\\n";
	for (size_t n = 0; n < arguments_.size(); n++) {
		const variable * argument = arguments_[n];
		sstrm << argument->debug_string() << " (" << argument->type().debug_string() << ")\\n";
	}

	return sstrm.str();
}

const std::string &
cfg::enter_node::argument_name(size_t index) const
{
	JLM_DEBUG_ASSERT(index < arguments_.size());

	return arguments_[index]->name();
}

const jive::base::type &
cfg::enter_node::argument_type(size_t index) const
{
	JLM_DEBUG_ASSERT(index < arguments_.size());

	return arguments_[index]->type();
}

const variable *
cfg::enter_node::argument(size_t index) const
{
	JLM_DEBUG_ASSERT(index < arguments_.size());

	return arguments_[index];
}

/* exit node */

cfg::exit_node::~exit_node() noexcept {}

cfg::exit_node::exit_node(jlm::cfg & cfg) noexcept
	: cfg_node(cfg)
{}

std::string
cfg::exit_node::debug_string() const
{
	std::stringstream sstrm;

	sstrm << this << " (EXIT)\\n";
	for (size_t n = 0; n < results_.size(); n++) {
		const jlm::variable * result = results_[n];
		sstrm << result->debug_string() << " (" << result->type().debug_string() << ")\\n";
	}

	return sstrm.str();
}

/* cfg */

cfg::cfg()
	: clg_node_(nullptr)
{
	create_enter_node();
	create_exit_node();
	enter_->add_outedge(exit_, 0);
}

cfg::cfg(jlm::clg_node & clg_node)
	: clg_node_(&clg_node)
{
	create_enter_node();
	create_exit_node();
	enter_->add_outedge(exit_, 0);
}

cfg::cfg(const cfg & c)
	: clg_node_(nullptr)
{
	std::unordered_map<cfg_node*,cfg_node*> node_map;
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it;
	for (it = c.nodes_.begin(); it != c.nodes_.end(); it++) {
		cfg_node * copy;
		cfg_node * node = (*it).get();
		if (node == c.enter()) {
			create_enter_node();
			copy = enter_;
		} else if (node == c.exit()) {
			create_exit_node();
			copy = exit_;
		} else
			copy = create_basic_block();

		node_map[node] = copy;
	}

	for (it = c.nodes_.begin(); it != c.nodes_.end(); it++) {
		cfg_node * node = (*it).get();
		cfg_node * copy = node_map[node];
		std::vector<cfg_edge*> edges = node->outedges();
		for (auto edge : edges)
			copy->add_outedge(node_map[edge->sink()], edge->index());
	}
}

void
cfg::create_enter_node()
{
	std::unique_ptr<cfg_node> enter(new enter_node(*this));
	enter_ = static_cast<enter_node*>(enter.get());
	nodes_.insert(std::move(enter));
	enter.release();
}

void
cfg::create_exit_node()
{
	std::unique_ptr<cfg_node> exit(new exit_node(*this));
	exit_ = static_cast<exit_node*>(exit.get());
	nodes_.insert(std::move(exit));
	exit.release();
}

basic_block *
cfg::create_basic_block()
{
	std::unique_ptr<cfg_node> bb(new basic_block(*this));
	basic_block * tmp = static_cast<basic_block*>(bb.get());
	nodes_.insert(std::move(bb));
	bb.release();
	return tmp;
}

const jlm::variable *
cfg::create_variable(const jive::base::type & type)
{
	std::stringstream sstr;
	sstr << "v" << variables_.size();
	std::unique_ptr<variable> variable(new jlm::variable(type, sstr.str()));
	jlm::variable * v = variable.get();
	variables_.insert(std::move(variable));
	variable.release();
	return v;
}

const jlm::variable *
cfg::create_variable(const jive::base::type & type, const std::string & name)
{
	std::unique_ptr<variable> variable(new jlm::variable(type, name));
	jlm::variable * v = variable.get();
	variables_.insert(std::move(variable));
	variable.release();
	return v;
}

std::vector<std::unordered_set<cfg_node*>>
cfg::find_sccs() const
{
	JLM_DEBUG_ASSERT(is_closed());

	std::vector<std::unordered_set<cfg_node*>> sccs;

	std::unordered_map<cfg_node*, std::pair<size_t,size_t>> map;
	std::vector<cfg_node*> node_stack;
	size_t index = 0;

	strongconnect(enter(), exit(), map, node_stack, index, sccs);

	return sccs;
}

bool
cfg::is_closed() const noexcept
{
	JLM_DEBUG_ASSERT(is_valid());

	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it;
	for (it = nodes_.begin(); it != nodes_.end(); it++) {
		cfg_node * node = (*it).get();
		if (node == enter())
			continue;

		if (node->no_predecessor())
			return false;
	}

	return true;
}

bool
cfg::is_linear() const noexcept
{
	JLM_DEBUG_ASSERT(is_closed());

	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it;
	for (it = nodes_.begin(); it != nodes_.end(); it++) {
		cfg_node * node = (*it).get();
		if (node == enter() || node == exit())
			continue;

		if (!node->single_successor() || !node->single_predecessor())
			return false;
	}

	return true;
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
	JLM_DEBUG_ASSERT(is_closed());

	cfg c(*this);
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it = c.nodes_.begin();
	while (it != c.nodes_.end()) {
		cfg_node * node = (*it).get();

		if (c.nodes_.size() == 2) {
			JLM_DEBUG_ASSERT(c.is_closed());
			return true;
		}

		if (node == c.enter() || node == c.exit()) {
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
	JLM_DEBUG_ASSERT(is_closed());

	cfg c(*this);
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it = c.nodes_.begin();
	while (it != c.nodes_.end()) {
		cfg_node * node = (*it).get();

		if (c.nodes_.size() == 2) {
			JLM_DEBUG_ASSERT(c.is_closed());
			return true;
		}

		if (node == c.enter() || node == c.exit()) {
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

bool
cfg::is_valid() const
{
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it;
	for (it = nodes_.begin(); it != nodes_.end(); it++) {
		cfg_node * node = (*it).get();
		if (node == exit()) {
			if (!node->no_successor())
				return false;
			continue;
		}

		if (node == enter()) {
			if (!node->no_predecessor())
				return false;
			if (!node->single_successor())
				return false;
			if (node->outedges()[0]->index() != 0)
				return false;
			continue;
		}

		if (node->no_successor())
			return false;

		/*
			Check whether all indices are 0 and in ascending order (uniqueness of indices)
		*/
		std::vector<cfg_edge*> edges = node->outedges();
		std::sort(edges.begin(), edges.end(),
			[](const cfg_edge * e1, const cfg_edge * e2)
			{ return e1->index() < e2->index(); });
		for (size_t n = 0; n < edges.size(); n++) {
			if (edges[n]->index() != n)
				return false;
		}

		/*
			Check whether the CFG is actually a graph and not a multigraph
		*/
		std::sort(edges.begin(), edges.end(),
			[](const cfg_edge * e1, const cfg_edge * e2)
			{ return e1->sink() < e2->sink(); });
		for (size_t n = 1; n < edges.size(); n++) {
			if (edges[n-1]->sink() == edges[n]->sink())
				return false;
		}
	}

	return true;
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
	JLM_DEBUG_ASSERT(is_valid());

	/* find all nodes that are dominated by the entry node */
	std::unordered_set<cfg_node*> to_visit({enter_});
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

	JLM_DEBUG_ASSERT(is_closed());
}

void
cfg::destruct_ssa()
{
	/* find all blocks containing phis */
	std::unordered_set<basic_block*> phi_blocks;
	for (auto it = nodes_.begin(); it != nodes_.end(); it++) {
		cfg_node * node = it->get();
		if (!dynamic_cast<basic_block*>(node))
			continue;

		basic_block * bb = static_cast<basic_block*>(node);
		if (!bb->tacs().empty() && dynamic_cast<const phi_op*>(&bb->tacs().front()->operation()))
			phi_blocks.insert(bb);
	}

	/* eliminate phis */
	for (auto phi_block : phi_blocks) {
		basic_block * ass_block = create_basic_block();

		std::list<const tac*> & tacs = phi_block->tacs();
		for (auto tac : tacs) {
			if (!dynamic_cast<const phi_op*>(&tac->operation()))
				break;

			const phi_op * phi = static_cast<const phi_op*>(&tac->operation());
			const variable * v = create_variable(phi->type());

			size_t n = 0;
			const variable * value = nullptr;
			std::list<cfg_edge*> edges = phi_block->inedges();
			for (auto it = edges.begin(); it != edges.end(); it++, n++) {
				basic_block * edge_block = static_cast<basic_block*>((*it)->split());

				value = assignment_tac(edge_block, v, tac->input(n));
			}
			assignment_tac(ass_block, tac->output(0), value);
		}

		phi_block->divert_inedges(ass_block);
		ass_block->add_outedge(phi_block, 0);

		/* remove phi TACs */
		while (!tacs.empty()) {
			if (!dynamic_cast<const phi_op*>(&tacs.front()->operation()))
				break;
			tacs.pop_front();
		}
	}
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
