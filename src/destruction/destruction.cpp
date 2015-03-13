/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/destruction/context.hpp>
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

#include <stack>

namespace jlm {

class dststate {
public:
	jlm::dstrct::predicate_stack pstack;
	jlm::dstrct::theta_stack tstack;
	jlm::dstrct::variable_map vmap;
};

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
	const jlm::dstrct::context & ctx)
{
	if (inedges.size() <= 1)
		return false;

	for (auto edge : inedges) {
		if (ctx.is_back_edge(edge))
			return false;
	}

	return true;
}

static bool
visit_branch_join(
	const jlm::frontend::basic_block * bb,
	const jlm::dstrct::context & ctx)
{
	for (auto edge : bb->inedges()) {
		if (!ctx.has_predicate_stack(edge))
			return false;
	}

	return true;
}

static bool
is_loop_entry(
	const std::list<jlm::frontend::cfg_edge*> & inedges,
	const jlm::dstrct::context & ctx)
{
	if (inedges.size() != 2)
		return false;

	jlm::frontend::cfg_edge * edge1 = inedges.front();
	jlm::frontend::cfg_edge * edge2 = *std::next(inedges.begin());
	return ctx.is_back_edge(edge1) || ctx.is_back_edge(edge2);
}

static bool
is_loop_exit(
	const std::vector<jlm::frontend::cfg_edge*> & outedges,
	const jlm::dstrct::context & ctx)
{
	if (outedges.size() != 2)
		return false;

	return ctx.is_back_edge(outedges[0]) || ctx.is_back_edge(outedges[1]);
}

static dststate
handle_branch_join(
	const std::list<jlm::frontend::cfg_edge*> & inedges,
	struct jive_graph * graph,
	jlm::dstrct::context & ctx)
{
	/* get predicate and new predicate stack */
	dstrct::predicate_stack pstack = ctx.lookup_predicate_stack(inedges.front());
	jive::output * predicate = pstack.top();
	for (auto edge : inedges)
		JLM_DEBUG_ASSERT(predicate == ctx.lookup_predicate_stack(edge).top());
	pstack.pop();

	/* check theta stack */
	dstrct::theta_stack tstack = ctx.lookup_theta_stack(inedges.front());
	for (auto edge : inedges) {
		JLM_DEBUG_ASSERT(tstack.size() == ctx.lookup_theta_stack(edge).size());
		if (!tstack.empty())
			JLM_DEBUG_ASSERT(tstack.top() == ctx.lookup_theta_stack(edge).top());
	}

	/* compute all variables needed in the gamma */
	std::unordered_set<const frontend::variable*> variables = ctx.lookup_demand(inedges.front());
	for (auto edge : inedges)
		JIVE_DEBUG_ASSERT(ctx.lookup_demand(edge) == variables);

	/* compute operand types for gamma */
	std::vector<const jive::base::type*> types;
	for (auto variable : variables)
		types.push_back(&variable->type());

	/* set up gamma operands */
	std::vector<std::vector<jive::output*>> alternatives;
	for (auto edge : inedges) {
		std::vector<jive::output*> values;
		for (auto variable : variables) {
			if (ctx.lookup_variable_map(edge->source()).has_value(variable)) {
				values.push_back(ctx.lookup_variable_map(edge->source()).lookup_value(variable));
			} else {
				values.push_back(create_undefined_value(variable->type(), graph));
			}
		}
		alternatives.push_back(values);
	}

	/* create gamma */
	std::vector<jive::output*> results = jive_gamma(predicate, types, alternatives);
	JLM_DEBUG_ASSERT(results.size() == variables.size());

	/* update variable map */
	size_t n = 0;
	dstrct::variable_map vmap;
	for (auto variable : variables)
		vmap.insert_value(variable, results[n++]);

	dststate state;
	state.pstack = pstack;
	state.tstack = tstack;
	state.vmap = vmap;

	return state;
}

static dststate
handle_loop_entry(
	const jlm::frontend::cfg_edge * entry_edge,
	struct jive_graph * graph,
	jlm::dstrct::context & ctx)
{
	dstrct::variable_map vmap = ctx.lookup_variable_map(entry_edge->source());
	dstrct::theta_stack tstack = ctx.lookup_theta_stack(entry_edge);
	dstrct::predicate_stack pstack = ctx.lookup_predicate_stack(entry_edge);

	/* create new theta environment and update variable map */
	dstrct::theta_env * tenv = ctx.create_theta_env(graph);
	std::unordered_set<const frontend::variable*> variables = ctx.lookup_demand(entry_edge);

	for (auto variable : variables) {
		jive::output * value = vmap.lookup_value(variable);
		jive_theta_loopvar lv = jive_theta_loopvar_enter(*tenv->theta(), value);
		tenv->insert_loopvar(variable, lv);
		vmap.replace_value(variable, lv.value);
	}
	tstack.push(tenv);

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
	jlm::dstrct::context & ctx)
{
	const std::list<jlm::frontend::cfg_edge*> & inedges = bb->inedges();

	if (is_loop_entry(inedges, ctx)) {
		jlm::frontend::cfg_edge * entry_edge = inedges.front();
		if (ctx.is_back_edge(entry_edge))
			entry_edge = *std::next(inedges.begin());

		return handle_loop_entry(entry_edge, graph, ctx);
	}

	if (is_branch_join(inedges, ctx))
		return handle_branch_join(inedges, graph, ctx);

	JLM_DEBUG_ASSERT(inedges.size() == 1);

	dststate state;
	state.vmap = ctx.lookup_variable_map(inedges.front()->source());
	state.pstack = ctx.lookup_predicate_stack(inedges.front());
	state.tstack = ctx.lookup_theta_stack(inedges.front());
	return state;
}

static void
handle_basic_block(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	dstrct::variable_map & vmap)
{
	for (auto tac : bb->tacs()) {
		if (dynamic_cast<const jlm::frontend::assignment_op*>(&tac->operation())) {
			vmap.insert_value(tac->output(0), vmap.lookup_value(tac->input(0)));
			continue;
		}

		std::vector<jive::output*> operands;
		for (size_t n = 0; n < tac->ninputs(); n++)
			operands.push_back(vmap.lookup_value(tac->input(n)));

		std::vector<jive::output *> results;
		results = jive_node_create_normalized(graph, tac->operation(), operands);

		JLM_DEBUG_ASSERT(results.size() == tac->noutputs());
		for (size_t n = 0; n < tac->noutputs(); n++)
			vmap.insert_value(tac->output(n), results[n]);
	}
}

static void
handle_loop_exit(
	const jlm::frontend::tac * match,
	struct jive_graph * graph,
	dstrct::variable_map & vmap,
	dstrct::theta_stack & tstack)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::match_op*>(&match->operation()));
	jive::output * predicate = vmap.lookup_value(match->output(0));

	dstrct::theta_env * tenv = tstack.poptop();
	std::vector<jive_theta_loopvar> loopvars;
	for (auto it = vmap.begin(); it != vmap.end(); it++) {
		jive_theta_loopvar lv;
		if (tenv->has_loopvar(it->first)) {
			lv = tenv->lookup_loopvar(it->first);
			jive_theta_loopvar_leave(*tenv->theta(), lv.gate, it->second);
		} else {
			lv = jive_theta_loopvar_enter(*tenv->theta(),
				create_undefined_value(it->first->type(), graph));
			tenv->insert_loopvar(it->first, lv);
			jive_theta_loopvar_leave(*tenv->theta(), lv.gate, it->second);
		}
		loopvars.push_back(lv);
	}

	jive_theta_end(*tenv->theta(), predicate, loopvars.size(), &loopvars[0]);

	size_t n = 0;
	JIVE_DEBUG_ASSERT(loopvars.size() == vmap.size());
	for (auto it = vmap.begin(); it != vmap.end(); it++, n++)
		tenv->replace_loopvar(it->first, loopvars[n]);

	for (auto it = vmap.begin(); it != vmap.end(); it++) {
		jive_theta_loopvar lv = tenv->lookup_loopvar(it->first);
		vmap.replace_value(it->first, lv.value);
	}
}

static void
handle_basic_block_exit(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	jlm::dstrct::context & ctx,
	dststate & state)
{
	const std::vector<jlm::frontend::cfg_edge*> & outedges = bb->outedges();

	const jlm::frontend::tac * tac;
	if (is_loop_exit(outedges, ctx)) {
		JLM_DEBUG_ASSERT(outedges.size() == 2);
		tac = static_cast<frontend::basic_block*>(outedges[0]->source())->tacs().back();
		JLM_DEBUG_ASSERT(tac->noutputs() == 1);

		handle_loop_exit(tac, graph, state.vmap, state.tstack);
		return;
	}

	/* handle branch split */
	if (outedges.size() > 1) {
		tac = static_cast<frontend::basic_block*>(outedges[0]->source())->tacs().back();
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::match_op*>(&tac->operation()));

		state.pstack.push(state.vmap.lookup_value(tac->output(0)));
		return;
	}
}

static void
convert_basic_block(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	jlm::dstrct::context & ctx)
{
	/* only process basic block if all incoming edges have already been visited */
	if (is_branch_join(bb->inedges(), ctx) && !visit_branch_join(bb, ctx))
		return;

	dststate state = handle_basic_block_entry(bb, graph, ctx);
	handle_basic_block(bb, graph, state.vmap);
	handle_basic_block_exit(bb, graph, ctx, state);

	ctx.insert_variable_map(bb, state.vmap);
	for (auto e : bb->outedges()) {
		if (ctx.is_back_edge(e))
			continue;

		ctx.insert_predicate_stack(e, state.pstack);
		ctx.insert_theta_stack(e, state.tstack);
		if (auto bb = dynamic_cast<jlm::frontend::basic_block *>(e->sink())) {
			convert_basic_block(bb, graph, ctx);
		}
	}
}

static void
convert_basic_blocks(
	const jlm::frontend::cfg * cfg,
	struct jive_graph * graph,
	jlm::dstrct::context & ctx)
{
	JLM_DEBUG_ASSERT(cfg->enter()->noutedges() == 1);
	const jlm::frontend::cfg_edge * edge = cfg->enter()->outedges()[0];

	ctx.insert_predicate_stack(edge, dstrct::predicate_stack());
	ctx.insert_theta_stack(edge, dstrct::theta_stack());

	const jlm::frontend::basic_block * bb;
	bb = static_cast<const jlm::frontend::basic_block*>(edge->sink());
	convert_basic_block(bb, graph, ctx);
}

static void
compute_demands(
	const jlm::frontend::cfg_node * node,
	dstrct::demand_map & dmap,
	const std::unordered_set<const jlm::frontend::cfg_edge*> & back_edges)
{
	/* check that all output edges are present */
	for (auto edge : node->outedges()) {
		if (back_edges.find(edge) != back_edges.end())
			continue;

		if (!dmap.exists(edge))
			return;
	}

	/* compute demand for node */
	std::unordered_set<const frontend::variable*> demands;
	for (auto edge : node->outedges()) {
		if (back_edges.find(edge) != back_edges.end())
			continue;

		demands.insert(dmap.lookup(edge).begin(), dmap.lookup(edge).end());
	}

	/* handle TACs */
	if (node->cfg()->is_exit(node)) {
		JIVE_DEBUG_ASSERT(demands.empty());
		for (size_t n = 0; n < node->cfg()->nresults(); n++)
			demands.insert(node->cfg()->result(n));
	} else if (node->cfg()->is_enter(node)) {
		/* nothing needs to be done */
	} else if (auto bb = dynamic_cast<const frontend::basic_block*>(node)) {
		for (auto it = bb->tacs().rbegin(); it != bb->tacs().rend(); it++) {
			const frontend::tac * tac = *it;
			for (size_t n = 0; n < tac->noutputs(); n++)
				demands.erase(tac->output(n));
			for (size_t n = 0; n < tac->ninputs(); n++)
				demands.insert(tac->input(n));
		}
	}	else
		JIVE_DEBUG_ASSERT(0);

	/* insert demands into demand map and process next node */
	for (auto edge : node->inedges()) {
		if (back_edges.find(edge) != back_edges.end())
			continue;
		dmap.insert(edge, demands);
		compute_demands(edge->source(), dmap, back_edges);
	}
}

static dstrct::demand_map
compute_demands(
	const jlm::frontend::cfg * cfg,
	const std::unordered_set<const frontend::cfg_edge*> & back_edges)
{
	dstrct::demand_map dmap;
	compute_demands(cfg->exit(), dmap, back_edges);
	return dmap;
}

static jive::output * 
convert_cfg(
	jlm::frontend::cfg * cfg,
	struct jive_graph * graph)
{
	jive_cfg_view(*cfg);
	cfg->destruct_ssa();
	jive_cfg_view(*cfg);
	const std::unordered_set<const frontend::cfg_edge*> back_edges = restructure(cfg);
	jive_cfg_view(*cfg);

	dstrct::demand_map dmap = compute_demands(cfg, back_edges);

	jlm::dstrct::context ctx(dmap, back_edges);

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

	jlm::dstrct::variable_map vmap;
	JLM_DEBUG_ASSERT(variables.size() == lambda->narguments);
	for (size_t n = 0; n < variables.size(); n++)
		vmap.insert_value(variables[n], lambda->arguments[n]);
	ctx.insert_variable_map(cfg->enter(), vmap);

	convert_basic_blocks(cfg, graph, ctx);

	JLM_DEBUG_ASSERT(cfg->exit()->ninedges() == 1);
	jlm::frontend::cfg_node * predecessor = cfg->exit()->inedges().front()->source();

	std::vector<jive::output*> results;
	std::vector<const jive::base::type*> result_types;
	for (size_t n = 0; n< cfg->nresults(); n++) {
		results.push_back(ctx.lookup_variable_map(predecessor).lookup_value(cfg->result(n)));
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
