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
	struct jive_region * region,
	jlm::dstrct::context & ctx)
{
	/* check theta stack */
	dstrct::theta_stack tstack = ctx.lookup_theta_stack(inedges.front());
	for (auto edge : inedges) {
		JLM_DEBUG_ASSERT(tstack.size() == ctx.lookup_theta_stack(edge).size());
		if (!tstack.empty())
			JLM_DEBUG_ASSERT(tstack.top() == ctx.lookup_theta_stack(edge).top());
	}

	/* compute all variables needed in the gamma */
	dstrct::variable_map vmap;
	for (auto edge : inedges)
		vmap.merge(ctx.lookup_variable_map(edge->source()));

	/* compute operand types for gamma */
	std::vector<const jive::base::type*> types;
	for (auto vpair : vmap)
		types.push_back(&vpair.first->type());

	jive::output * predicate = nullptr;
	std::vector<std::vector<jive::output*>> alternatives(inedges.size());
	for (auto edge : inedges) {
		/* get predicate and index */
		size_t index = ctx.lookup_predicate_stack(edge).top().second;
		jive::output * p = ctx.lookup_predicate_stack(edge).top().first;
		if (!predicate) predicate = p;
		JIVE_DEBUG_ASSERT(predicate == p);
		JIVE_DEBUG_ASSERT(index < inedges.size());

		/* set up gamma operands */
		std::vector<jive::output*> values;
		for (auto vpair : vmap) {
			const jlm::frontend::variable * variable = vpair.first;
			if (ctx.lookup_variable_map(edge->source()).has_value(variable)) {
				values.push_back(ctx.lookup_variable_map(edge->source()).lookup_value(variable));
			} else {
				values.push_back(create_undefined_value(variable->type(), region->graph));
			}
		}
		alternatives[index] = values;
	}
	dstrct::predicate_stack pstack = ctx.lookup_predicate_stack(inedges.front());
	pstack.pop();

	/* create gamma */
	std::vector<jive::output*> results = jive_gamma(predicate, types, alternatives);
	JLM_DEBUG_ASSERT(results.size() == vmap.size());

	/* update variable map */
	size_t n = 0;
	for (auto vpair : vmap)
		vmap.replace_value(vpair.first, results[n++]);

	dststate state;
	state.pstack = pstack;
	state.tstack = tstack;
	state.vmap = vmap;

	return state;
}

static dststate
handle_loop_entry(
	const jlm::frontend::cfg_edge * entry_edge,
	struct jive_region * region,
	jlm::dstrct::context & ctx)
{
	dstrct::variable_map vmap = ctx.lookup_variable_map(entry_edge->source());
	dstrct::theta_stack tstack = ctx.lookup_theta_stack(entry_edge);
	dstrct::predicate_stack pstack = ctx.lookup_predicate_stack(entry_edge);

	if (!tstack.empty())
		region = tstack.top()->theta()->region;

	/* create new theta environment and update variable map */
	dstrct::theta_env * tenv = ctx.create_theta_env(region);
	for (auto vpair : vmap) {
		jive_theta_loopvar lv = jive_theta_loopvar_enter(*tenv->theta(), vpair.second);
		tenv->insert_loopvar(vpair.first, lv);
		vmap.replace_value(vpair.first, lv.value);
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
	struct jive_region * region,
	jlm::dstrct::context & ctx)
{
	const std::list<jlm::frontend::cfg_edge*> & inedges = bb->inedges();

	if (is_loop_entry(inedges, ctx)) {
		jlm::frontend::cfg_edge * entry_edge = inedges.front();
		if (ctx.is_back_edge(entry_edge))
			entry_edge = *std::next(inedges.begin());

		return handle_loop_entry(entry_edge, region, ctx);
	}

	if (is_branch_join(inedges, ctx))
		return handle_branch_join(inedges, region, ctx);

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
	struct jive_region * region,
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
		results = jive_node_create_normalized(region->graph, tac->operation(), operands);

		JLM_DEBUG_ASSERT(results.size() == tac->noutputs());
		for (size_t n = 0; n < tac->noutputs(); n++)
			vmap.insert_value(tac->output(n), results[n]);
	}
}

static void
handle_loop_exit(
	const jlm::frontend::tac * match,
	struct jive_region * region,
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
				create_undefined_value(it->first->type(), region->graph));
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
	struct jive_region * region,
	jlm::dstrct::context & ctx,
	dststate & state)
{
	const std::vector<jlm::frontend::cfg_edge*> & outedges = bb->outedges();

	/* handle loop exit */
	const jlm::frontend::tac * tac;
	if (is_loop_exit(outedges, ctx)) {
		JLM_DEBUG_ASSERT(outedges.size() == 2);
		tac = static_cast<frontend::basic_block*>(outedges[0]->source())->tacs().back();
		JLM_DEBUG_ASSERT(tac->noutputs() == 1);

		handle_loop_exit(tac, region, state.vmap, state.tstack);

		for (auto edge : outedges) {
			if (ctx.is_back_edge(edge))
				continue;

			ctx.insert_predicate_stack(edge, state.pstack);
		}
		return;
	}

	/* handle branch split */
	if (outedges.size() > 1) {
		tac = static_cast<frontend::basic_block*>(outedges[0]->source())->tacs().back();
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::match_op*>(&tac->operation()));

		for (auto edge : outedges) {
			if (ctx.is_back_edge(edge))
				continue;

			dstrct::predicate_stack pstack = state.pstack;
			pstack.push(state.vmap.lookup_value(tac->output(0)), edge->index());
			ctx.insert_predicate_stack(edge, pstack);
		}
		return;
	}

	/* handle outgoing edge */
	JIVE_DEBUG_ASSERT(outedges.size() == 1);
	ctx.insert_predicate_stack(outedges[0], state.pstack);
}

static void
convert_basic_block(
	const jlm::frontend::basic_block * bb,
	struct jive_region * region,
	jlm::dstrct::context & ctx)
{
	/* only process basic block if all incoming edges have already been visited */
	if (is_branch_join(bb->inedges(), ctx) && !visit_branch_join(bb, ctx))
		return;

	dststate state = handle_basic_block_entry(bb, region, ctx);
	handle_basic_block(bb, region, state.vmap);
	handle_basic_block_exit(bb, region, ctx, state);

	ctx.insert_variable_map(bb, state.vmap);
	for (auto e : bb->outedges()) {
		if (ctx.is_back_edge(e))
			continue;

		ctx.insert_theta_stack(e, state.tstack);
		if (auto bb = dynamic_cast<jlm::frontend::basic_block *>(e->sink())) {
			convert_basic_block(bb, region, ctx);
		}
	}
}

static void
convert_basic_blocks(
	const jlm::frontend::cfg * cfg,
	struct jive_region * region,
	jlm::dstrct::context & ctx)
{
	JLM_DEBUG_ASSERT(cfg->enter()->noutedges() == 1);
	const jlm::frontend::cfg_edge * edge = cfg->enter()->outedges()[0];

	ctx.insert_predicate_stack(edge, dstrct::predicate_stack());
	ctx.insert_theta_stack(edge, dstrct::theta_stack());

	const jlm::frontend::basic_block * bb;
	bb = static_cast<const jlm::frontend::basic_block*>(edge->sink());
	convert_basic_block(bb, region, ctx);
}

static jive::output * 
convert_cfg(
	jlm::frontend::cfg * cfg,
	struct jive_region * region)
{
//	jive_cfg_view(*cfg);
	cfg->destruct_ssa();
//	jive_cfg_view(*cfg);
	jlm::dstrct::context ctx(restructure(cfg));
//	jive_cfg_view(*cfg);


	std::vector<const jlm::frontend::variable*> variables;
	std::vector<const char*> argument_names;
	std::vector<const jive::base::type*> argument_types;
	for (size_t n = 0; n < cfg->narguments(); n++) {
		variables.push_back(cfg->argument(n));
		argument_names.push_back(cfg->argument(n)->name().c_str());
		argument_types.push_back(&cfg->argument(n)->type());
	}

	struct jive_lambda * lambda = jive_lambda_begin(region, variables.size(),
		&argument_types[0], &argument_names[0]);

	jlm::dstrct::variable_map vmap;
	JLM_DEBUG_ASSERT(variables.size() == lambda->narguments);
	for (size_t n = 0; n < variables.size(); n++)
		vmap.insert_value(variables[n], lambda->arguments[n]);
	ctx.insert_variable_map(cfg->enter(), vmap);

	convert_basic_blocks(cfg, lambda->region, ctx);

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

	jive::output * f = convert_cfg(clg_node->cfg(), graph->root_region);
	/* FIXME: we export everything right now */
	jive_graph_export(graph, f, clg_node->name());
	return f;
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
