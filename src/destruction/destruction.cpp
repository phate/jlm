/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/destruction/destruction.hpp>
#include <jlm/destruction/restructuring.hpp>

#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/tac/assignment.hpp>
#include <jlm/IR/tac/tac.hpp>

#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/function/fctlambda.h>
#include <jive/vsdg/basetype.h>
#include <jive/vsdg/control.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/operators/match.h>
#include <jive/vsdg/theta.h>

//to be removed
#include <jive/view.h>

#include <stack>

namespace jlm {

class theta {
public:
	jive_theta theta;
	std::unordered_map<const jlm::frontend::variable*, jive_theta_loopvar> loopvars;
};

}

typedef std::stack<jive::output*> predicate_stack;
typedef std::stack<jlm::theta> theta_stack;

typedef std::unordered_map<const jlm::frontend::variable*, jive::output*> variable_map;
typedef std::unordered_map<const jlm::frontend::cfg_node*, variable_map> node_map;
typedef std::unordered_map<const jlm::frontend::cfg_edge*, predicate_stack> predicate_map;
typedef std::unordered_map<const jlm::frontend::cfg_edge*, theta_stack> theta_map;

class dststate {
public:
	predicate_stack pstack;
	theta_stack tstack;
	variable_map vmap;
};

namespace jlm {

static jive::output *
create_undefined_value(const jive::base::type & type, struct jive_graph * graph)
{
	if (auto t = dynamic_cast<const jive::bits::type*>(&type))
		return jive_bitconstant_undefined(graph, t->nbits());

	/* FIXME: temporary solutation */
	if (dynamic_cast<const jive::ctl::type*>(&type))
		return jive_control_constant(graph, 2, 0);

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static bool
is_branch_join(
	const std::list<jlm::frontend::cfg_edge*> & inedges,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	if (inedges.size() <= 1)
		return false;

	for (auto edge : inedges) {
		if (back_edges.find(edge) != back_edges.end())
			return false;
	}

	return true;
}

static bool
visit_branch_join(const jlm::frontend::basic_block * bb, predicate_map & pmap)
{
	for (auto edge : bb->inedges()) {
		if (pmap.find(edge) == pmap.end())
			return false;
	}

	return true;
}

static bool
is_loop_entry(
	const std::list<jlm::frontend::cfg_edge*> & inedges,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	if (inedges.size() != 2)
		return false;

	jlm::frontend::cfg_edge * edge1 = inedges.front();
	jlm::frontend::cfg_edge * edge2 = *std::next(inedges.begin());
	return back_edges.find(edge1) != back_edges.end() || back_edges.find(edge2) != back_edges.end();
}

static bool
is_loop_exit(
	const std::vector<jlm::frontend::cfg_edge*> & outedges,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	if (outedges.size() != 2)
		return false;

	jlm::frontend::cfg_edge * edge1 = outedges[0];
	jlm::frontend::cfg_edge * edge2 = outedges[1];
	return back_edges.find(edge1) != back_edges.end() || back_edges.find(edge2) != back_edges.end();
}

static dststate
handle_branch_join(
	const std::list<jlm::frontend::cfg_edge*> & inedges,
	struct jive_graph * graph,
	const node_map & nmap,
	const predicate_map & pmap,
	const theta_map & tmap)
{
	variable_map vmap;
	predicate_stack pstack = pmap.at(inedges.front());

	JLM_DEBUG_ASSERT(pstack.size() != 0);
	jive::output * predicate = pstack.top();
	for (auto edge : inedges)
		JLM_DEBUG_ASSERT(predicate == pmap.at(edge).top());
	pstack.pop();

	for (auto edge : inedges)
		vmap.insert(nmap.at(edge->source()).begin(), nmap.at(edge->source()).end());

	std::vector<const jive::base::type*> types;
	for (auto vpair : vmap)
		types.push_back(&vpair.first->type());

	std::vector<std::vector<jive::output*>> alternatives;
	for (auto edge : inedges) {
		std::vector<jive::output*> values;
		for (auto vpair : vmap) {
			const jlm::frontend::variable * variable = vpair.first;
			if (nmap.at(edge->source()).find(variable) != nmap.at(edge->source()).end()) {
				jive::output * value = nmap.at(edge->source()).at(variable);
				JLM_DEBUG_ASSERT(value != nullptr);
				values.push_back(value);
			} else {
				jive::output * undef = create_undefined_value(variable->type(), graph);
				JLM_DEBUG_ASSERT(undef != nullptr);
				values.push_back(undef);
			}
		}
		alternatives.push_back(values);
	}

	std::vector<jive::output*> results = jive_gamma(predicate, types, alternatives);
	JLM_DEBUG_ASSERT(results.size() == vmap.size());

	size_t n = 0;
	for (auto variable : vmap)
		vmap[variable.first] = results[n++];

	dststate state;
	state.pstack = pstack;
	state.tstack = tmap.at(inedges.front());
	state.vmap = vmap;

	return state;
}

static dststate
handle_loop_entry(
	const jlm::frontend::cfg_edge * entry_edge,
	struct jive_graph * graph,
	const node_map & nmap,
	const predicate_map & pmap,
	const theta_map & tmap)
{
	theta_stack tstack = tmap.at(entry_edge);
	predicate_stack pstack = pmap.at(entry_edge);
	variable_map vmap(nmap.at(entry_edge->source()).begin(), nmap.at(entry_edge->source()).end());

	jlm::theta theta;
	theta.theta = jive_theta_begin(graph);
	for (auto vpair : vmap) {
		jive_theta_loopvar loopvar = jive_theta_loopvar_enter(theta.theta, vpair.second);
		JLM_DEBUG_ASSERT(loopvar.value);
		vmap[vpair.first] = loopvar.value;
		theta.loopvars[vpair.first] = loopvar;
	}
	tstack.push(theta);

	dststate state;
	state.pstack = pstack;
	state.tstack = tstack;
	state.vmap = vmap;

	return state;
}

static dststate
handle_basic_block_entry(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	const node_map & nmap,
	const predicate_map & pmap,
	const theta_map & tmap,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	const std::list<jlm::frontend::cfg_edge*> & inedges = bb->inedges();

	if (is_loop_entry(inedges, back_edges)) {
		jlm::frontend::cfg_edge * entry_edge = inedges.front();
		jlm::frontend::cfg_edge * back_edge = *std::next(inedges.begin());
		if (back_edges.find(entry_edge) != back_edges.end()) {
			jlm::frontend::cfg_edge * tmp = entry_edge;
			entry_edge = back_edge;
			back_edge = tmp;
		}

		return handle_loop_entry(entry_edge, graph, nmap, pmap, tmap);
	}

	if (is_branch_join(inedges, back_edges))
		return handle_branch_join(inedges, graph, nmap, pmap, tmap);

	JLM_DEBUG_ASSERT(inedges.size() == 1);

	dststate state;
	state.vmap = nmap.at(inedges.front()->source());
	state.pstack = pmap.at(inedges.front());
	state.tstack = tmap.at(inedges.front());
	return state;
}

static void
handle_basic_block(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	variable_map & vmap)
{
	std::list<const jlm::frontend::tac*> tacs = bb->tacs();
	for (auto tac : tacs) {
		if (dynamic_cast<const jlm::frontend::assignment_op*>(&tac->operation())) {
			JLM_DEBUG_ASSERT(vmap.find(tac->input(0)) != vmap.end());
			vmap[tac->output(0)] = vmap[tac->input(0)];
			continue;
		}

		std::vector<jive::output*> operands;
		for (size_t n = 0; n < tac->ninputs(); n++) {
			JLM_DEBUG_ASSERT(vmap.find(tac->input(n)) != vmap.end());
			operands.push_back(vmap[tac->input(n)]);
		}

		std::vector<jive::output *> results;
		results = jive_node_create_normalized(graph, tac->operation(), operands);

		JLM_DEBUG_ASSERT(results.size() == tac->noutputs());
		for (size_t n = 0; n < tac->noutputs(); n++) {
			JLM_DEBUG_ASSERT(results[n] != nullptr);
			vmap[tac->output(n)] = results[n];
		}
	}
}

static void
handle_loop_exit(
	const jlm::frontend::variable * pv,
	struct jive_graph * graph,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges,
	variable_map & vmap,
	std::stack<theta> & tstack)
{
	JLM_DEBUG_ASSERT(tstack.size() != 0);
	jlm::theta theta = tstack.top();
	tstack.pop();

	std::vector<jive_theta_loopvar> tmp;
	for (auto lvpair : theta.loopvars) {
		jive_theta_loopvar_leave(theta.theta, lvpair.second.gate, vmap[lvpair.first]);
		tmp.push_back(lvpair.second);
	}

	jive::output * predicate = vmap[pv];
	jive_theta_end(theta.theta, predicate, tmp.size(), &tmp[0]);

	size_t n = 0;
	for (auto it = theta.loopvars.begin(); it != theta.loopvars.end(); it++) {
		JLM_DEBUG_ASSERT(tmp[n].value != nullptr);
		vmap[it->first] = tmp[n++].value;
	}
}

static void
handle_basic_block_exit(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges,
	dststate & state)
{
	const std::vector<jlm::frontend::cfg_edge*> & outedges = bb->outedges();

	const jlm::frontend::tac * tac;
	if (is_loop_exit(outedges, back_edges)) {
		JLM_DEBUG_ASSERT(outedges.size() == 2);
		tac = static_cast<frontend::basic_block*>(outedges[0]->source())->tacs().back();
		JLM_DEBUG_ASSERT(tac->noutputs() == 1);

		handle_loop_exit(tac->output(0), graph, back_edges, state.vmap, state.tstack);
		return;
	}

	/* handle branch split */
	if (outedges.size() > 1) {
		tac = static_cast<frontend::basic_block*>(outedges[0]->source())->tacs().back();
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::match_op*>(&tac->operation()));
		JLM_DEBUG_ASSERT(state.vmap[tac->output(0)]);

		state.pstack.push(state.vmap[tac->output(0)]);
		return;
	}
}

static void
convert_basic_block(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	node_map & nmap,
	predicate_map & pmap,
	theta_map & tmap,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	/* only process basic block if all incoming edges have already been visited */
	if (is_branch_join(bb->inedges(), back_edges) && !visit_branch_join(bb, pmap))
		return;

	dststate state = handle_basic_block_entry(bb, graph, nmap, pmap, tmap, back_edges);
	handle_basic_block(bb, graph, state.vmap);
	handle_basic_block_exit(bb, graph, back_edges, state);

	nmap[bb] = state.vmap;
	for (auto e : bb->outedges()) {
		if (back_edges.find(e) != back_edges.end())
			continue;

		pmap[e] = state.pstack;
		tmap[e] = state.tstack;
		if (auto bb = dynamic_cast<jlm::frontend::basic_block *>(e->sink())) {
			convert_basic_block(bb, graph, nmap, pmap, tmap, back_edges);
		}
	}
}

static void
convert_basic_blocks(
	const jlm::frontend::cfg * cfg,
	struct jive_graph * graph,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges,
	node_map & nmap)
{
	theta_map tmap;
	predicate_map pmap;
	pmap[cfg->enter()->outedges()[0]] = std::stack<jive::output*>();
	tmap[cfg->enter()->outedges()[0]] = std::stack<jlm::theta>();

	const jlm::frontend::basic_block * bb;
	bb = static_cast<const jlm::frontend::basic_block*>(cfg->enter()->outedges()[0]->sink());

	convert_basic_block(bb, graph, nmap, pmap, tmap, back_edges);
}

static jive::output * 
convert_cfg(
	jlm::frontend::cfg * cfg,
	struct jive_graph * graph)
{
	jive_cfg_view(*cfg);
	cfg->destruct_ssa();
	jive_cfg_view(*cfg);
	std::unordered_set<jlm::frontend::cfg_edge*> back_edges = restructure(cfg);
	jive_cfg_view(*cfg);


	std::vector<const jlm::frontend::variable*> variables;
	std::vector<const char*> argument_names;
	std::vector<const jive::base::type*> argument_types;
	for (size_t n = 0; n < cfg->narguments(); n++) {
		variables.push_back(cfg->argument(n));
		argument_names.push_back(cfg->argument(n)->name().c_str());
		argument_types.push_back(&cfg->argument(n)->type());
	}

	struct jive_lambda * lambda = jive_lambda_begin(graph, variables.size(),
		&argument_types[0], &argument_names[0]);

	variable_map vmap;
	JLM_DEBUG_ASSERT(variables.size() == lambda->narguments);
	for (size_t n = 0; n < variables.size(); n++)
		vmap[variables[n]] = lambda->arguments[n];

	node_map nmap;
	nmap[cfg->enter()] = vmap;
	JLM_DEBUG_ASSERT(cfg->enter()->noutedges() == 1);

	convert_basic_blocks(cfg, graph, back_edges, nmap);

	JLM_DEBUG_ASSERT(cfg->exit()->ninedges() == 1);
	jlm::frontend::cfg_node * predecessor = cfg->exit()->inedges().front()->source();

	std::vector<jive::output*> results;
	std::vector<const jive::base::type*> result_types;
	for (size_t n = 0; n< cfg->nresults(); n++) {
		results.push_back(nmap[predecessor][cfg->result(n)]);
		result_types.push_back(&cfg->result(n)->type());
	}

	return jive_lambda_end(lambda, cfg->nresults(), &result_types[0], &results[0]);
}

static jive::output *
construct_lambda(struct jive_graph * graph, const jlm::frontend::clg_node * clg_node)
{
	//FIXME: check whether cfg_node has a CFG

	return convert_cfg(clg_node->cfg(), graph);
}


static void
handle_scc(struct jive_graph * graph, std::unordered_set<const jlm::frontend::clg_node*> & scc)
{
	if (scc.size() == 1 && !(*scc.begin())->is_selfrecursive()) {
		construct_lambda(graph, *scc.begin());
	} else {
		JLM_DEBUG_ASSERT(0);
		/* create phi */
	}
}

struct jive_graph *
construct_rvsdg(const jlm::frontend::clg & clg)
{
	struct ::jive_graph * graph = jive_graph_create();	

	std::vector<std::unordered_set<const jlm::frontend::clg_node*>> sccs = clg.find_sccs();
	for (auto scc : sccs)
		handle_scc(graph, scc);

	return graph;
}

}
