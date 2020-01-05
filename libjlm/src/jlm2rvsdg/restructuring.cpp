/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/cfg-node.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators/operators.hpp>

#include <jive/rvsdg/control.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace jlm {

struct scc_structure {
	typedef std::unordered_set<jlm::cfg_node*>::const_iterator const_node_iterator;
	typedef std::unordered_set<jlm::cfg_edge*>::const_iterator const_edge_iterator;

	inline
	scc_structure(
		const std::unordered_set<jlm::cfg_node*> & enodes,
		const std::unordered_set<jlm::cfg_node*> & xnodes,
		const std::unordered_set<jlm::cfg_edge*> & eedges,
		const std::unordered_set<jlm::cfg_edge*> & redges,
		const std::unordered_set<jlm::cfg_edge*> & xedges)
	: enodes_(enodes)
	, xnodes_(xnodes)
	, eedges_(eedges)
	, redges_(redges)
	, xedges_(xedges)
	{}

	inline size_t
	nenodes() const noexcept
	{
		return enodes_.size();
	}

	inline size_t
	nxnodes() const noexcept
	{
		return xnodes_.size();
	}

	inline size_t
	needges() const noexcept
	{
		return eedges_.size();
	}

	inline size_t
	nredges() const noexcept
	{
		return redges_.size();
	}

	inline size_t
	nxedges() const noexcept
	{
		return xedges_.size();
	}

	inline const_node_iterator
	begin_enodes() const
	{
		return enodes_.begin();
	}

	inline const_node_iterator
	end_enodes() const
	{
		return enodes_.end();
	}

	inline const_node_iterator
	begin_xnodes() const
	{
		return xnodes_.begin();
	}

	inline const_node_iterator
	end_xnodes() const
	{
		return xnodes_.end();
	}

	inline const_edge_iterator
	begin_eedges() const
	{
		return eedges_.begin();
	}

	inline const_edge_iterator
	end_eedges() const
	{
		return eedges_.end();
	}

	inline const_edge_iterator
	begin_redges() const
	{
		return redges_.begin();
	}

	inline const_edge_iterator
	end_redges() const
	{
		return redges_.end();
	}

	inline const_edge_iterator
	begin_xedges() const
	{
		return xedges_.begin();
	}

	inline const_edge_iterator
	end_xedges() const
	{
		return xedges_.end();
	}

	std::unordered_set<jlm::cfg_node*> enodes_;
	std::unordered_set<jlm::cfg_node*> xnodes_;
	std::unordered_set<jlm::cfg_edge*> eedges_;
	std::unordered_set<jlm::cfg_edge*> redges_;
	std::unordered_set<jlm::cfg_edge*> xedges_;
};

static inline bool
is_tcloop(const scc_structure & s)
{
	return s.nenodes() == 1 && s.nredges() == 1 && s.nxedges() == 1
			&& (*s.begin_redges())->source() == (*s.begin_xedges())->source();
}

struct tcloop {
	inline
	tcloop(
		jlm::cfg_node * entry,
		basic_block * i,
		basic_block * r)
	: ne(entry)
	, insert(i)
	, replacement(r)
	{}

	jlm::cfg_node * ne;
	basic_block * insert;
	basic_block * replacement;
};

static inline tcloop
extract_tcloop(jlm::cfg_node * ne, jlm::cfg_node * nx)
{
	JLM_DEBUG_ASSERT(nx->noutedges() == 2);
	auto & cfg = ne->cfg();

	auto er = nx->outedge(0);
	auto ex = nx->outedge(1);
	if (er->sink() != ne) {
		er = nx->outedge(1);
		ex = nx->outedge(0);
	}
	JLM_DEBUG_ASSERT(er->sink() == ne);

	auto exsink = basic_block::create(cfg);
	auto replacement = basic_block::create(cfg);
	ne->divert_inedges(replacement);
	replacement->add_outedge(ex->sink());
	ex->divert(exsink);
	er->divert(ne);

	return tcloop(ne, exsink, replacement);
}

static inline void
reinsert_tcloop(const tcloop & l)
{
	JLM_DEBUG_ASSERT(l.insert->ninedges() == 1);
	JLM_DEBUG_ASSERT(l.replacement->noutedges() == 1);
	auto & cfg = l.ne->cfg();

	l.replacement->divert_inedges(l.ne);
	l.insert->divert_inedges(l.replacement->outedge(0)->sink());

	cfg.remove_node(l.insert);
	cfg.remove_node(l.replacement);
}

static scc_structure
find_scc_structure(const std::unordered_set<jlm::cfg_node*> & scc)
{
	std::unordered_set<jlm::cfg_edge*> eedges;
	std::unordered_set<jlm::cfg_edge*> redges;
	std::unordered_set<jlm::cfg_edge*> xedges;
	std::unordered_set<jlm::cfg_node*> enodes;
	std::unordered_set<jlm::cfg_node*> xnodes;

	for (auto node : scc) {
		for (auto it = node->begin_inedges(); it != node->end_inedges(); it++) {
			if (scc.find((*it)->source()) == scc.end()) {
				eedges.insert(*it);
				if (enodes.find(node) == enodes.end())
					enodes.insert(node);
			}
		}

		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			if (scc.find(it->sink()) == scc.end()) {
				xedges.insert(it.edge());
				if (xnodes.find(it->sink()) == xnodes.end())
					xnodes.insert(it->sink());
			}
		}
	}

	for (auto node : scc) {
		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			if (enodes.find(it->sink()) != enodes.end())
				redges.insert(it.edge());
		}
	}

	return scc_structure(enodes, xnodes, eedges, redges, xedges);
}

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
		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			jlm::cfg_node * successor = it->sink();
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

static std::vector<std::unordered_set<cfg_node*>>
find_sccs(jlm::cfg_node * enter, jlm::cfg_node * exit)
{
	std::vector<std::unordered_set<cfg_node*>> sccs;

	std::unordered_map<cfg_node*, std::pair<size_t,size_t>> map;
	std::vector<cfg_node*> node_stack;
	size_t index = 0;

	strongconnect(enter, exit, map, node_stack, index, sccs);

	return sccs;
}

static inline const variable *
create_pvariable(const jive::ctltype & type, ipgraph_module & im)
{
	static size_t c = 0;
	return im.create_variable(type, strfmt("#p", c++, "#"));
}

static inline const variable *
create_qvariable(const jive::ctltype & type, ipgraph_module & im)
{
	static size_t c = 0;
	return im.create_variable(type, strfmt("#q", c++, "#"));
}

static const variable *
create_tvariable(const jive::ctltype & type, ipgraph_module & im)
{
	static size_t c = 0;
	return im.create_variable(type, strfmt("#t", c++, "#"));
}

static inline const variable *
create_rvariable(ipgraph_module & im)
{
	static size_t c = 0;
	jive::ctltype type(2);
	return im.create_variable(type, strfmt("#r", c++, "#"));
}

static inline void
append_branch(basic_block * bb, const variable * operand)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::ctltype*>(&operand->type()));
	auto nalternatives = static_cast<const jive::ctltype*>(&operand->type())->nalternatives();
	bb->append_last(create_branch_tac(nalternatives, operand));
}

static inline void
append_constant(basic_block * bb, const variable * result, size_t value)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::ctltype*>(&result->type()));
	auto nalternatives = static_cast<const jive::ctltype*>(&result->type())->nalternatives();

	jive::ctlconstant_op op(jive::ctlvalue_repr(value, nalternatives));
	bb->append_last(tac::create(op, {}, {result}));
}

static inline void
restructure_loop_entry(const scc_structure & s, basic_block * new_ne, const variable * ev)
{
	size_t n = 0;
	std::unordered_map<jlm::cfg_node*, size_t> indices;
	for (auto it = s.begin_enodes(); it != s.end_enodes(); it++, n++) {
		new_ne->add_outedge(*it);
		indices[*it] = n;
	}

	if (ev) append_branch(new_ne, ev);

	for (auto it = s.begin_eedges(); it != s.end_eedges(); it++) {
		auto os = (*it)->sink();
		(*it)->divert(new_ne);
		if (ev) append_constant((*it)->split(), ev, indices[os]);
	}
}

static inline void
restructure_loop_exit(
	const scc_structure & s,
	basic_block * new_nr,
	basic_block * new_nx,
	const variable * rv,
	const variable * xv)
{
	size_t n = 0;
	std::unordered_map<jlm::cfg_node*, size_t> indices;
	for (auto it = s.begin_xnodes(); it != s.end_xnodes(); it++, n++) {
		new_nx->add_outedge(*it);
		indices[*it] = n;
	}

	if (xv) append_branch(new_nx, xv);

	for (auto it = s.begin_xedges(); it != s.end_xedges(); it++) {
		auto os = (*it)->sink();
		(*it)->divert(new_nr);
		auto bb = (*it)->split();
		if (xv) append_constant(bb, xv, indices[os]);
		append_constant(bb, rv, 0);
	}
}

static inline void
restructure_loop_repetition(
	const scc_structure & s,
	jlm::cfg_node * new_nr,
	jlm::cfg_node * new_nx,
	const variable * ev,
	const variable * rv)
{
	size_t n = 0;
	std::unordered_map<jlm::cfg_node*, size_t> indices;
	for (auto it = s.begin_enodes(); it != s.end_enodes(); it++, n++)
		indices[*it] = n;

	for (auto it = s.begin_redges(); it != s.end_redges(); it++) {
		auto os = (*it)->sink();
		(*it)->divert(new_nr);
		auto bb = (*it)->split();
		if (ev) append_constant(bb, ev, indices[os]);
		append_constant(bb, rv, 1);
	}
}

static void
restructure(jlm::cfg_node*, jlm::cfg_node*, std::vector<tcloop>&);

static void
restructure_loops(jlm::cfg_node * entry, jlm::cfg_node * exit, std::vector<tcloop> & loops)
{
	if (entry == exit)
		return;

	auto & cfg = entry->cfg();
	auto & module = cfg.module();

	auto sccs = find_sccs(entry, exit);
	for (auto scc : sccs) {
		auto s = find_scc_structure(scc);

		if (is_tcloop(s)) {
			restructure(*s.begin_enodes(), (*s.begin_xedges())->source(), loops);
			loops.push_back(extract_tcloop(*s.begin_enodes(), (*s.begin_xedges())->source()));
			continue;
		}

		auto ev = s.nenodes() > 1 ? create_tvariable(jive::ctltype(s.nenodes()), module) : nullptr;
		auto rv = create_rvariable(module);
		auto xv = s.nxnodes() > 1 ? create_qvariable(jive::ctltype(s.nxnodes()), module) : nullptr;
		auto new_ne = basic_block::create(cfg);
		auto new_nr = basic_block::create(cfg);
		auto new_nx = basic_block::create(cfg);
		new_nr->add_outedge(new_nx);
		new_nr->add_outedge(new_ne);
		append_branch(new_nr, rv);

		restructure_loop_entry(s, new_ne, ev);
		restructure_loop_exit(s, new_nr, new_nx, rv, xv);
		restructure_loop_repetition(s, new_nr, new_nr, ev, rv);

		restructure(new_ne, new_nr, loops);
		loops.push_back(extract_tcloop(new_ne, new_nr));
	}
}

static jlm::cfg_node *
find_head_branch(jlm::cfg_node * start, jlm::cfg_node * end)
{
	do {
		if (start->is_branch() || start == end)
			break;

		start = start->outedge(0)->sink();
	} while (1);

	return start;
}

static std::unordered_set<jlm::cfg_node*>
find_dominator_graph(const jlm::cfg_edge * edge)
{
	std::unordered_set<jlm::cfg_node*> nodes;
	std::unordered_set<const jlm::cfg_edge*> edges({edge});

	std::deque<jlm::cfg_node*> to_visit(1, edge->sink());
	while (to_visit.size() != 0) {
		jlm::cfg_node * node = to_visit.front(); to_visit.pop_front();
		if (nodes.find(node) != nodes.end())
			continue;

		bool accept = true;
		for (auto it = node->begin_inedges(); it != node->end_inedges(); it++) {
			if (edges.find(*it) == edges.end()) {
				accept = false;
				break;
			}
		}

		if (accept) {
			nodes.insert(node);
			for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
				edges.insert(it.edge());
				to_visit.push_back(it->sink());
			}
		}
	}

	return nodes;
}

struct continuation {
	std::unordered_set<jlm::cfg_node*> points;
	std::unordered_map<jlm::cfg_edge*, std::unordered_set<jlm::cfg_edge*>> edges;
};

static inline continuation
compute_continuation(jlm::cfg_node * hb)
{
	JLM_DEBUG_ASSERT(hb->noutedges() > 1);

	std::unordered_map<jlm::cfg_edge*, std::unordered_set<jlm::cfg_node*>> dgraphs;
	for (auto it = hb->begin_outedges(); it != hb->end_outedges(); it++)
		dgraphs[it.edge()] = find_dominator_graph(it.edge());

	continuation c;
	for (auto it = hb->begin_outedges(); it != hb->end_outedges(); it++) {
		auto & dgraph = dgraphs[it.edge()];
		if (dgraph.empty()) {
			c.edges[it.edge()].insert(it.edge());
			c.points.insert(it.edge()->sink());
			continue;
		}

		for (const auto & node : dgraph) {
			for (auto it2 = node->begin_outedges(); it2 != node->end_outedges(); it2++) {
				if (dgraph.find(it2->sink()) == dgraph.end()) {
					c.edges[it.edge()].insert(it2.edge());
					c.points.insert(it2->sink());
				}
			}
		}
	}

	return c;
}

static inline void
restructure_branches(jlm::cfg_node * entry, jlm::cfg_node * exit)
{
	auto & cfg = entry->cfg();
	auto & module = cfg.module();

	auto hb = find_head_branch(entry, exit);
	if (hb == exit) return;

	auto c = compute_continuation(hb);
	JLM_DEBUG_ASSERT(!c.points.empty());

	if (c.points.size() == 1) {
		auto cpoint = *c.points.begin();
		for (auto it = hb->begin_outedges(); it != hb->end_outedges(); it++) {
			auto cedges = c.edges[it.edge()];

			/* empty branch subgraph */
			if (it->sink() == cpoint) {
				it->split();
				continue;
			}

			/* only one continuation edge */
			if (cedges.size() == 1) {
				auto e = *cedges.begin();
				JLM_DEBUG_ASSERT(e != it.edge());
				restructure_branches(it->sink(), e->source());
				continue;
			}

			/* more than one continuation edge */
			auto null = basic_block::create(cfg);
			null->add_outedge(cpoint);
			for (const auto & e : cedges)
				e->divert(null);
			restructure_branches(it->sink(), null);
		}

		/* restructure tail subgraph */
		restructure_branches(cpoint, exit);
		return;
	}

	/* insert new continuation point */
	auto p = create_pvariable(jive::ctltype(c.points.size()), module);
	auto cn = basic_block::create(cfg);
	append_branch(cn, p);
	std::unordered_map<cfg_node*, size_t> indices;
	for (const auto & cp : c.points) {
		cn->add_outedge(cp);
		indices.insert({cp, indices.size()});
	}

	/* restructure branch subgraphs */
	for (auto it = hb->begin_outedges(); it != hb->end_outedges(); it++) {
		auto cedges = c.edges[it.edge()];

		auto null = basic_block::create(cfg);
		null->add_outedge(cn);
		for (const auto & e : cedges) {
			auto bb = basic_block::create(cfg);
			append_constant(bb, p, indices[e->sink()]);
			bb->add_outedge(null);
			e->divert(bb);
		}

		restructure_branches(it->sink(), null);
	}

	/* restructure tail subgraph */
	restructure_branches(cn, exit);
}

void
restructure_loops(jlm::cfg * cfg)
{
	JLM_DEBUG_ASSERT(is_closed(*cfg));

	std::vector<tcloop> loops;
	restructure_loops(cfg->entry(), cfg->exit(), loops);

	for (const auto & l : loops)
		reinsert_tcloop(l);
}

void
restructure_branches(jlm::cfg * cfg)
{
	JLM_DEBUG_ASSERT(is_acyclic(*cfg));
	restructure_branches(cfg->entry(), cfg->exit());
	JLM_DEBUG_ASSERT(is_proper_structured(*cfg));
}

static inline void
restructure(jlm::cfg_node * entry, jlm::cfg_node * exit, std::vector<tcloop> & tcloops)
{
	restructure_loops(entry, exit, tcloops);
	restructure_branches(entry, exit);
}

void
restructure(jlm::cfg * cfg)
{
	JLM_DEBUG_ASSERT(is_closed(*cfg));

	std::vector<tcloop> tcloops;
	restructure(cfg->entry(), cfg->exit(), tcloops);

	for (const auto & l : tcloops)
		reinsert_tcloop(l);

	JLM_DEBUG_ASSERT(is_proper_structured(*cfg));
}

}
