/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/destruction/context.hpp>
#include <jlm/destruction/destruction.hpp>
#include <jlm/destruction/restructuring.hpp>

#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/module.hpp>
#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

#include <jive/arch/address.h>
#include <jive/arch/addresstype.h>
#include <jive/arch/dataobject.h>
#include <jive/arch/memlayout-simple.h>
#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/float.h>
#include <jive/types/function.h>
#include <jive/vsdg/basetype.h>
#include <jive/vsdg/control.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/operators/match.h>
#include <jive/vsdg/phi.h>
#include <jive/vsdg/region.h>

#include <cmath>
#include <stack>

namespace jlm {
#if 0
class dststate {
public:
	jlm::dstrct::predicate_stack pstack;
	jlm::dstrct::theta_stack tstack;
	jlm::dstrct::variable_map vmap;
};

static jive::oport *
create_undefined_value(const jive::base::type & type, jive::graph * graph)
{
	if (auto t = dynamic_cast<const jive::bits::type*>(&type))
		return jive_bitconstant_undefined(graph->root(), t->nbits());

	/* FIXME: temporary solutation */
	if (auto t = dynamic_cast<const jive::ctl::type*>(&type))
		return jive_control_constant(graph->root(), t->nalternatives(), 0);

	/* FIXME */
	if (dynamic_cast<const jive::addr::type*>(&type))
		return jive::address::constant(graph, jive::address::value_repr(0));

	/* FIXME */
	if (dynamic_cast<const jive::flt::type*>(&type))
		return jive_fltconstant(graph->root(), std::nan(""));

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static inline bool
is_back_edge(
	cfg_edge * edge,
	const std::unordered_set<const cfg_edge*> & back_edges)
{
	return back_edges.find(edge) != back_edges.end();
}

static bool
is_branch_join(
	const std::list<jlm::cfg_edge*> & inedges,
	const std::unordered_set<const cfg_edge*> & back_edges)
{
	if (inedges.size() <= 1)
		return false;

	for (auto edge : inedges) {
		if (is_back_edge(edge, back_edges))
			return false;
	}

	return true;
}

static bool
visit_branch_join(
	const jlm::basic_block * bb,
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
	const std::list<jlm::cfg_edge*> & inedges,
	const std::unordered_set<const cfg_edge*> & back_edges)
{
	if (inedges.size() != 2)
		return false;

	jlm::cfg_edge * edge1 = inedges.front();
	jlm::cfg_edge * edge2 = *std::next(inedges.begin());
	return is_back_edge(edge1, back_edges) || is_back_edge(edge2, back_edges);
}

static bool
is_loop_exit(
	const std::vector<jlm::cfg_edge*> & outedges,
	const std::unordered_set<const cfg_edge*> & back_edges)
{
	if (outedges.size() != 2)
		return false;

	return is_back_edge(outedges[0], back_edges) || is_back_edge(outedges[1], back_edges);
}

static dststate
handle_branch_join(
	const std::list<jlm::cfg_edge*> & inedges,
	jive::region * region,
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
	std::vector<std::vector<jive::oport*>> alternatives(inedges.size());
	for (auto edge : inedges) {
		/* get predicate and index */
		size_t index = ctx.lookup_predicate_stack(edge).top().second;
		jive::output * p = ctx.lookup_predicate_stack(edge).top().first;
		if (!predicate) predicate = p;
		JLM_DEBUG_ASSERT(predicate == p);
		JLM_DEBUG_ASSERT(index < inedges.size());

		/* set up gamma operands */
		std::vector<jive::oport*> values;
		for (auto vpair : vmap) {
			const jlm::variable * variable = vpair.first;
			if (ctx.lookup_variable_map(edge->source()).has_value(variable)) {
				values.push_back(ctx.lookup_variable_map(edge->source()).lookup_value(variable));
			} else {
				values.push_back(create_undefined_value(variable->type(), region->graph()));
			}
		}
		alternatives[index] = values;
	}
	dstrct::predicate_stack pstack = ctx.lookup_predicate_stack(inedges.front());
	pstack.pop();

	/* create gamma */
	/*FIXME: broken */
	std::vector<jive::output*> results;// = jive_gamma(predicate, types, alternatives);
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
	const jlm::cfg_edge * entry_edge,
	jive::region * region,
	jlm::dstrct::context & ctx)
{
	dstrct::variable_map vmap = ctx.lookup_variable_map(entry_edge->source());
	dstrct::theta_stack tstack = ctx.lookup_theta_stack(entry_edge);
	dstrct::predicate_stack pstack = ctx.lookup_predicate_stack(entry_edge);
#if 0
	if (!tstack.empty())
		/* FIXME: broken */
		region = nullptr;//tstack.top()->theta()->region;

	/* create new theta environment and update variable map */
	/* FIXME: broken */
	dstrct::theta_env * tenv = nullptr;//ctx.create_theta_env(region);
	for (auto vpair : vmap) {
		jive_theta_loopvar lv = jive_theta_loopvar_enter(*tenv->theta(), vpair.second);
		tenv->insert_loopvar(vpair.first, lv);
		vmap.replace_value(vpair.first, lv.value);
	}
	tstack.push(tenv);
#endif
	dststate state;
	state.pstack = pstack;
	state.tstack = tstack;
	state.vmap = vmap;

	return state;
}

static dststate
handle_basic_block_entry(
	const jlm::basic_block * bb,
	jive::region * region,
	jlm::dstrct::context & ctx,
	const std::unordered_set<const cfg_edge*> & back_edges)
{
	const std::list<jlm::cfg_edge*> & inedges = bb->inedges();

	if (is_loop_entry(inedges, back_edges)) {
		jlm::cfg_edge * entry_edge = inedges.front();
		if (is_back_edge(entry_edge, back_edges))
			entry_edge = *std::next(inedges.begin());

		return handle_loop_entry(entry_edge, region, ctx);
	}

	if (is_branch_join(inedges, back_edges))
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
	const jlm::basic_block * bb,
	jive::region * region,
	dstrct::context & ctx,
	dstrct::variable_map & vmap)
{
	for (auto tac : bb->tacs()) {
		if (dynamic_cast<const jlm::assignment_op*>(&tac->operation())) {
			vmap.insert_value(tac->output(0), vmap.lookup_value(tac->input(0)));
			continue;
		}

		if (dynamic_cast<const jlm::select_op*>(&tac->operation())) {
			//jive::oport * predicate = jive::ctl::match(1, {{0,0}}, 1, 2,
			//	vmap.lookup_value(tac->input(0)));

			//jive::output * tv = vmap.lookup_value(tac->input(1));
			//jive::output * fv = vmap.lookup_value(tac->input(2));
			/* FIXME: broken */
			jive::output * result = nullptr;//jive_gamma(predicate, {&select_op->type()}, {{fv}, {tv}})[0];
			vmap.insert_value(tac->output(0), result);

			continue;
		}

		std::vector<jive::output*> operands;
		for (size_t n = 0; n < tac->ninputs(); n++) {
			jive::output * value = nullptr;
			if (vmap.has_value(tac->input(n)))
				value = vmap.lookup_value(tac->input(n));
			else
				value = ctx.globals().lookup_value(tac->input(n));

			operands.push_back(value);
		}

		std::vector<jive::output *> results;
		//results = jive_node_create_normalized(region->graph, tac->operation(), operands);

		JLM_DEBUG_ASSERT(results.size() == tac->noutputs());
		for (size_t n = 0; n < tac->noutputs(); n++)
			vmap.insert_value(tac->output(n), results[n]);
	}
}

static void
handle_loop_exit(
	const jlm::tac * match,
	jive::region * region,
	dstrct::variable_map & vmap,
	dstrct::theta_stack & tstack)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::match_op*>(&match->operation()));
//	jive::output * predicate = vmap.lookup_value(match->output(0));
#if 0
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
	JLM_DEBUG_ASSERT(loopvars.size() == vmap.size());
	for (auto it = vmap.begin(); it != vmap.end(); it++, n++)
		tenv->replace_loopvar(it->first, loopvars[n]);

	for (auto it = vmap.begin(); it != vmap.end(); it++) {
		jive_theta_loopvar lv = tenv->lookup_loopvar(it->first);
		vmap.replace_value(it->first, lv.value);
	}
#endif
}

static void
convert_basic_block(
	const jlm::basic_block * bb,
	jive::region * region,
	jlm::dstrct::context & ctx,
	const std::unordered_set<const cfg_edge*> & back_edges)
{
	/* only process basic block if all incoming edges have already been visited */
	if (is_branch_join(bb->inedges(), back_edges) && !visit_branch_join(bb, ctx))
		return;


	dststate state = handle_basic_block_entry(bb, region, ctx, back_edges);
	handle_basic_block(bb, region, ctx, state.vmap);


	const std::vector<jlm::cfg_edge*> & outedges = bb->outedges();

	/* handle loop exit */
	const jlm::tac * tac;
	if (is_loop_exit(outedges, back_edges)) {
		JLM_DEBUG_ASSERT(outedges.size() == 2);
		tac = static_cast<basic_block*>(outedges[0]->source())->tacs().back();
		JLM_DEBUG_ASSERT(tac->noutputs() == 1);

		handle_loop_exit(tac, region, state.vmap, state.tstack);
	}

	/* update context and convert next basic blocks */
	ctx.insert_variable_map(bb, state.vmap);
	for (auto edge : bb->outedges()) {
		if (is_back_edge(edge, back_edges))
			continue;

		dstrct::predicate_stack pstack = state.pstack;
		if (!is_loop_exit(outedges, back_edges) && outedges.size() > 1) {
			tac = bb->tacs().back();
			JLM_DEBUG_ASSERT(dynamic_cast<const jive::match_op*>(&tac->operation()));
			pstack.push(state.vmap.lookup_value(tac->output(0)), edge->index());
		}

		ctx.insert_predicate_stack(edge, pstack);
		ctx.insert_theta_stack(edge, state.tstack);
		if (auto bb = dynamic_cast<jlm::basic_block *>(edge->sink()))
			convert_basic_block(bb, region, ctx, back_edges);
	}
}

static void
convert_basic_blocks(
	const jlm::cfg * cfg,
	jive::region * region,
	jlm::dstrct::context & ctx,
	const std::unordered_set<const cfg_edge*> & back_edges)
{
	JLM_DEBUG_ASSERT(cfg->enter()->noutedges() == 1);
	const jlm::cfg_edge * edge = cfg->enter()->outedges()[0];

	ctx.insert_predicate_stack(edge, dstrct::predicate_stack());
	ctx.insert_theta_stack(edge, dstrct::theta_stack());

	const jlm::basic_block * bb;
	bb = static_cast<const jlm::basic_block*>(edge->sink());
	convert_basic_block(bb, region, ctx, back_edges);
}
#endif

static jive::output * 
convert_cfg(
	jlm::cfg * cfg,
	jive::region * region,
	dstrct::context & ctx)
{
//	jive_cfg_view(*cfg);
	cfg->destruct_ssa();
//	jive_cfg_view(*cfg);
	const std::unordered_set<const cfg_edge*> back_edges = restructure(cfg);
//	jive_cfg_view(*cfg);
#if 0
	std::vector<const jlm::variable*> variables;
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

	convert_basic_blocks(cfg, lambda->region, ctx, back_edges);

	JLM_DEBUG_ASSERT(cfg->exit()->ninedges() == 1);
	jlm::cfg_node * predecessor = cfg->exit()->inedges().front()->source();

	std::vector<jive::output*> results;
	std::vector<const jive::base::type*> result_types;
	for (size_t n = 0; n< cfg->nresults(); n++) {
		results.push_back(ctx.lookup_variable_map(predecessor).lookup_value(cfg->result(n)));
		result_types.push_back(&cfg->result(n)->type());
	}
#endif 
	return nullptr; //jive_lambda_end(lambda, cfg->nresults(), &result_types[0], &results[0]);
}

static jive::oport *
construct_lambda(
	const clg_node * function,
	jive::region * region,
	dstrct::context & ctx)
{
	if (function->cfg() == nullptr)
		return region->graph()->import(function->type(), function->name());

	return convert_cfg(function->cfg(), region, ctx);
}


static void
handle_scc(
	const std::unordered_set<const jlm::clg_node*> & scc,
	jive::graph * graph,
	dstrct::context & ctx)
{
	if (scc.size() == 1 && !(*scc.begin())->is_selfrecursive()) {
		const jlm::clg_node * function = *scc.begin();
		auto lambda = construct_lambda(function, graph->root(), ctx);
		ctx.globals().insert_value(function, lambda);
		if (function->exported())
			graph->export_port(lambda, function->name());
	} else {
		jive::phi_builder pb;
		pb.begin(graph->root());

		std::vector<std::shared_ptr<jive::recvar>> recvars;
		for (const auto & f : scc) {
			auto rv = pb.add_recvar(f->type());
			ctx.globals().insert_value(f, rv->value());
			recvars.push_back(rv);
		}

		size_t n = 0;
		for (auto it = scc.begin(); it != scc.end(); it++, n++) {
			auto lambda = construct_lambda(*it, pb.region(), ctx);
			recvars[n]->set_value(lambda);
		}
		pb.end();

		n = 0;
		for (auto it = scc.begin(); it != scc.end(); it++, n++) {
			ctx.globals().replace_value(*it, recvars[n]->value());
			if ((*it)->exported())
				graph->export_port(recvars[n]->value(), (*it)->name());
		}
	}
}

#if 0
static jive::output*
convert_expression(const expr & e, jive_graph * graph)
{
	std::vector<jive::output*> operands;
	for (size_t n = 0; n < e.noperands(); n++)
		operands.push_back(convert_expression(e.operand(n), graph));

	return jive_node_create_normalized(graph, e.operation(), operands)[0];
}

static dstrct::variable_map
convert_global_variables(const module & m, jive_graph * graph)
{
	jlm::dstrct::variable_map vmap;
	jive::memlayout_mapper_simple mapper(4);
	for (auto it = m.begin(); it != m.end(); it++) {
		jive::output * data = jive_dataobj(convert_expression(*(it->second), graph), &mapper);
		vmap.insert_value(it->first, data);
		if (it->first->exported())
			jive_graph_export(graph, data, it->first->name());
	}

	return vmap;
}
#endif

std::unique_ptr<jive::graph>
construct_rvsdg(const module & m)
{
	auto rvsdg = std::make_unique<jive::graph>();
/*
	dstrct::variable_map globals = convert_global_variables(m, graph);
*/
	dstrct::variable_map globals;

	dstrct::context ctx(globals);
	auto sccs = m.clg().find_sccs();
	for (const auto & scc : sccs)
		handle_scc(scc, rvsdg.get(), ctx);

	return std::move(rvsdg);
}

}
