/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace jlm {

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
	JLM_ASSERT(nx->noutedges() == 2);
	auto & cfg = ne->cfg();

	auto er = nx->outedge(0);
	auto ex = nx->outedge(1);
	if (er->sink() != ne) {
		er = nx->outedge(1);
		ex = nx->outedge(0);
	}
	JLM_ASSERT(er->sink() == ne);

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
	JLM_ASSERT(l.insert->ninedges() == 1);
	JLM_ASSERT(l.replacement->noutedges() == 1);
	auto & cfg = l.ne->cfg();

	l.replacement->divert_inedges(l.ne);
	l.insert->divert_inedges(l.replacement->outedge(0)->sink());

	cfg.remove_node(l.insert);
	cfg.remove_node(l.replacement);
}

static const tacvariable *
create_pvariable(
	basic_block & bb,
	const jive::ctltype & type)
{
	static size_t c = 0;
	auto name = strfmt("#p", c++, "#");
	return bb.insert_before_branch(UndefValueOperation::Create(type, name))->result(0);
}

static const tacvariable *
create_qvariable(
	basic_block & bb,
	const jive::ctltype & type)
{
	static size_t c = 0;
	auto name = strfmt("#q", c++, "#");
	return bb.append_last(UndefValueOperation::Create(type, name))->result(0);
}

static const tacvariable *
create_tvariable(
	basic_block & bb,
	const jive::ctltype & type)
{
	static size_t c = 0;
	auto name = strfmt("#q", c++, "#");
	return bb.insert_before_branch(UndefValueOperation::Create(type, name))->result(0);
}

static const tacvariable *
create_rvariable(basic_block & bb)
{
	static size_t c = 0;
	auto name = strfmt("#r", c++, "#");

	jive::ctltype type(2);
	return bb.append_last(UndefValueOperation::Create(type, name))->result(0);
}

static inline void
append_branch(basic_block * bb, const variable * operand)
{
	JLM_ASSERT(dynamic_cast<const jive::ctltype*>(&operand->type()));
	auto nalternatives = static_cast<const jive::ctltype*>(&operand->type())->nalternatives();
	bb->append_last(branch_op::create(nalternatives, operand));
}

static inline void
append_constant(
	basic_block * bb,
	const tacvariable * result,
	size_t value)
{
	JLM_ASSERT(dynamic_cast<const jive::ctltype*>(&result->type()));
	auto nalternatives = static_cast<const jive::ctltype*>(&result->type())->nalternatives();

	jive::ctlconstant_op op(jive::ctlvalue_repr(value, nalternatives));
	bb->append_last(tac::create(op, {}));
	bb->append_last(assignment_op::create(bb->last()->result(0), result));
}

static inline void
restructure_loop_entry(
	const sccstructure & s,
	basic_block * new_ne,
	const tacvariable * ev)
{
	size_t n = 0;
	std::unordered_map<jlm::cfg_node*, size_t> indices;
	for (auto & node : s.enodes()) {
		new_ne->add_outedge(node);
		indices[node] = n++;
	}

	if (ev) append_branch(new_ne, ev);

	for (auto & edge : s.eedges()) {
		auto os = edge->sink();
		edge->divert(new_ne);
		if (ev) append_constant(edge->split(), ev, indices[os]);
	}
}

static inline void
restructure_loop_exit(
	const sccstructure & s,
	basic_block * new_nr,
	basic_block * new_nx,
	cfg_node * exit,
	const tacvariable * rv,
	const tacvariable * xv)
{
	/*
		It could be that an SCC has no exit edge. This can arise when the input CFG contains a
		statically detectable endless loop, e.g., entry -> basic block  exit. Note the missing
		                                                   ^_________|
		edge to the exit node.

		Such CFGs do not play well with our restructuring algorithm, as the exit node does not
		post-dominate the basic block. We circumvent this problem by inserting an additional
		edge from the newly created exit basic block of the loop to the exit of the SESE region.
		This edge is never taken at runtime, but fixes the CFGs structure at compile-time such
		that we can create an RVSDG.
	*/
	if (s.nxedges() == 0) {
		new_nx->add_outedge(exit);
		return;
	}

	size_t n = 0;
	std::unordered_map<jlm::cfg_node*, size_t> indices;
	for (auto & node : s.xnodes()) {
		new_nx->add_outedge(node);
		indices[node] = n++;
	}

	if (xv) append_branch(new_nx, xv);

	for (auto & edge : s.xedges()) {
		auto os = edge->sink();
		edge->divert(new_nr);
		auto bb = edge->split();
		if (xv) append_constant(bb, xv, indices[os]);
		append_constant(bb, rv, 0);
	}
}

static inline void
restructure_loop_repetition(
	const sccstructure & s,
	jlm::cfg_node * new_nr,
	jlm::cfg_node * new_nx,
	const tacvariable * ev,
	const tacvariable * rv)
{
	size_t n = 0;
	std::unordered_map<jlm::cfg_node*, size_t> indices;
	for (auto & node : s.enodes())
		indices[node] = n++;

	for (auto & edge : s.redges()) {
		auto os = edge->sink();
		edge->divert(new_nr);
		auto bb = edge->split();
		if (ev) append_constant(bb, ev, indices[os]);
		append_constant(bb, rv, 1);
	}
}

static basic_block *
find_tvariable_bb(cfg_node * node)
{
	if (auto bb = dynamic_cast<basic_block*>(node))
		return bb;

	auto sink = node->outedge(0)->sink();
	JLM_ASSERT(is<basic_block>(sink));

	return static_cast<basic_block*>(sink);
}

static void
restructure(jlm::cfg_node*, jlm::cfg_node*, std::vector<tcloop>&);

static void
restructure_loops(jlm::cfg_node * entry, jlm::cfg_node * exit, std::vector<tcloop> & loops)
{
	if (entry == exit)
		return;

	auto & cfg = entry->cfg();

	auto sccs = find_sccs(entry, exit);
	for (auto & scc : sccs) {
		auto sccstruct = sccstructure::create(scc);

		if (sccstruct->is_tcloop()) {
			auto tcloop_entry = *sccstruct->enodes().begin();
			auto tcloop_exit = (*sccstruct->xedges().begin())->source();
			restructure(tcloop_entry, tcloop_exit, loops);
			loops.push_back(extract_tcloop(tcloop_entry, tcloop_exit));
			continue;
		}

		auto new_ne = basic_block::create(cfg);
		auto new_nr = basic_block::create(cfg);
		auto new_nx = basic_block::create(cfg);
		new_nr->add_outedge(new_nx);
		new_nr->add_outedge(new_ne);

		const tacvariable * ev = nullptr;
		if (sccstruct->nenodes() > 1) {
			auto bb = find_tvariable_bb(entry);
			ev = create_tvariable(*bb, jive::ctltype(sccstruct->nenodes()));
		}

		auto rv = create_rvariable(*new_ne);

		const tacvariable * xv = nullptr;
		if (sccstruct->nxnodes() > 1)
			xv = create_qvariable(*new_ne, jive::ctltype(sccstruct->nxnodes()));

		append_branch(new_nr, rv);

		restructure_loop_entry(*sccstruct, new_ne, ev);
		restructure_loop_exit(*sccstruct, new_nr, new_nx, exit, rv, xv);
		restructure_loop_repetition(*sccstruct, new_nr, new_nr, ev, rv);

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
		for (auto & inedge : node->inedges()) {
			if (edges.find(inedge) == edges.end()) {
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
	JLM_ASSERT(hb->noutedges() > 1);

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

	auto hb = find_head_branch(entry, exit);
	if (hb == exit) return;

	JLM_ASSERT(is<basic_block>(hb));
	auto & hbb = *static_cast<basic_block*>(hb);

	auto c = compute_continuation(hb);
	JLM_ASSERT(!c.points.empty());

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
				JLM_ASSERT(e != it.edge());
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
	auto p = create_pvariable(hbb, jive::ctltype(c.points.size()));
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
RestructureLoops(jlm::cfg * cfg)
{
	JLM_ASSERT(is_closed(*cfg));

	std::vector<tcloop> loops;
	restructure_loops(cfg->entry(), cfg->exit(), loops);

	for (const auto & l : loops)
		reinsert_tcloop(l);
}

void
RestructureBranches(jlm::cfg * cfg)
{
	JLM_ASSERT(is_acyclic(*cfg));
	restructure_branches(cfg->entry(), cfg->exit());
	JLM_ASSERT(is_proper_structured(*cfg));
}

static inline void
restructure(jlm::cfg_node * entry, jlm::cfg_node * exit, std::vector<tcloop> & tcloops)
{
	restructure_loops(entry, exit, tcloops);
	restructure_branches(entry, exit);
}

void
RestructureControlFlow(jlm::cfg * cfg)
{
	JLM_ASSERT(is_closed(*cfg));

	std::vector<tcloop> tcloops;
	restructure(cfg->entry(), cfg->exit(), tcloops);

	for (const auto & l : tcloops)
		reinsert_tcloop(l);

	JLM_ASSERT(is_proper_structured(*cfg));
}

}
