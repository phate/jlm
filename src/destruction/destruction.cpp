/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/destruction/destruction.hpp>
#include <jlm/destruction/restructuring.hpp>

#include <jlm/IR/aggregation/aggregation.hpp>
#include <jlm/IR/aggregation/annotation.hpp>
#include <jlm/IR/aggregation/node.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/module.hpp>
#include <jlm/IR/operators.hpp>
#include <jlm/IR/ssa.hpp>
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
#include <jive/vsdg/theta.h>

#include <cmath>
#include <stack>

namespace jlm {

typedef std::unordered_map<const variable*, jive::oport*> vmap;

class scoped_vmap final {
public:
	inline
	~scoped_vmap()
	{
		pop_vmap();
		JLM_DEBUG_ASSERT(nvmaps() == 0);
	}

	inline
	scoped_vmap(const jlm::module & module)
	: module_(module)
	{
		push_vmap();
	}

	inline size_t
	nvmaps() const noexcept
	{
		return vmaps_.size();
	}

	inline jlm::vmap &
	vmap(size_t n) noexcept
	{
		JLM_DEBUG_ASSERT(n < nvmaps());
		return vmaps_[n];
	}

	inline jlm::vmap &
	last() noexcept
	{
		JLM_DEBUG_ASSERT(nvmaps() > 0);
		return vmap(nvmaps()-1);
	}

	inline jlm::vmap &
	push_vmap()
	{
		vmaps_.push_back(jlm::vmap());
		return vmaps_.back();
	}

	inline void
	pop_vmap()
	{
		vmaps_.pop_back();
	}

	const jlm::module &
	module() const noexcept
	{
		return module_;
	}

private:
	const jlm::module & module_;
	std::vector<jlm::vmap> vmaps_;
};

static void
convert_assignment(const jlm::tac & tac, jive::region * region, scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_assignment_op(tac.operation()));
	JLM_DEBUG_ASSERT(tac.ninputs() == 1 && tac.noutputs() == 1);
	svmap.last()[tac.output(0)] = svmap.last()[tac.input(0)];
}

static void
convert_select(const jlm::tac & tac, jive::region * region, scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_select_op(tac.operation()));
	JLM_DEBUG_ASSERT(tac.ninputs() == 3 && tac.noutputs() == 1);

	auto op = jive::match_op(1, {{1, 0}}, 1, 2);
	auto predicate = jive::create_normalized(region, op, {svmap.last()[tac.input(0)]})[0];

	jive::gamma_builder gb;
	gb.begin(predicate);
	auto ev1 = gb.add_entryvar(svmap.last()[tac.input(1)]);
	auto ev2 = gb.add_entryvar(svmap.last()[tac.input(2)]);
	auto ex = gb.add_exitvar({ev1->argument(0), ev2->argument(1)});
	svmap.last()[tac.output(0)] = ex->output();
	gb.end();
}

static void
convert_tac(const jlm::tac & tac, jive::region * region, scoped_vmap & svmap)
{
	static std::unordered_map<
		std::type_index,
		std::function<void(const jlm::tac&, jive::region*, scoped_vmap&)>
	> map({
	  {std::type_index(typeid(assignment_op)), convert_assignment}
	, {std::type_index(typeid(select_op)), convert_select}
	});

	if (map.find(std::type_index(typeid(tac.operation()))) != map.end())
		return map[std::type_index(typeid(tac.operation()))](tac, region, svmap);

	std::vector<jive::oport*> operands;
	for (size_t n = 0; n < tac.ninputs(); n++) {
		JLM_DEBUG_ASSERT(svmap.last().find(tac.input(n)) != svmap.last().end());
		operands.push_back(svmap.last()[tac.input(n)]);
	}

	auto results = jive::create_normalized(region, tac.operation(), operands);

	JLM_DEBUG_ASSERT(results.size() == tac.noutputs());
	for (size_t n = 0; n < tac.noutputs(); n++)
		svmap.last()[tac.output(n)] = results[n];
}

static void
convert_basic_block(
	const basic_block & bb,
	jive::region * region,
	scoped_vmap & svmap)
{
	for (const auto & tac: bb)
		convert_tac(*tac, region, svmap);
}

static void
convert_node(
	const agg::node & node,
	const agg::demand_map & dm,
	jive::region * region,
	scoped_vmap & svmap);

static void
convert_entry_node(
	const agg::node & node,
	const agg::demand_map & dm,
	jive::region * region,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_entry_structure(node.structure()));
	auto ea = static_cast<const agg::entry*>(&node.structure())->attribute();

	JLM_DEBUG_ASSERT(ea.narguments() == region->narguments());
	for (size_t n = 0; n < ea.narguments(); n++)
		svmap.last()[ea.argument(n)] = region->argument(n);
}

static void
convert_exit_node(
	const agg::node & node,
	const agg::demand_map & dm,
	jive::region * region,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_exit_structure(node.structure()));
	auto xa = static_cast<const agg::exit*>(&node.structure())->attribute();

	std::vector<jive::oport*> results;
	for (size_t n = 0; n < xa.nresults(); n++) {
		JLM_DEBUG_ASSERT(svmap.last().find(xa.result(n)) != svmap.last().end());
		results.push_back(svmap.last()[xa.result(n)]);
	}
}

static void
convert_block_node(
	const agg::node & node,
	const agg::demand_map & dm,
	jive::region * region,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_block_structure(node.structure()));
	auto bb = static_cast<const agg::block*>(&node.structure())->basic_block();
	convert_basic_block(bb, region, svmap);
}

static void
convert_linear_node(
	const agg::node & node,
	const agg::demand_map & dm,
	jive::region * region,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_linear_structure(node.structure()));

	for (const auto & child : node)
		convert_node(child, dm, region, svmap);
}

static void
convert_branch_node(
	const agg::node & node,
	const agg::demand_map & dm,
	jive::region * region,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_branch_structure(node.structure()));
	auto branch = static_cast<const agg::branch*>(&node.structure());

	convert_basic_block(branch->split(), region, svmap);

	JLM_DEBUG_ASSERT(branch->split().last()->noutputs() == 1);
	auto predicate = svmap.last()[branch->split().last()->output(0)];
	jive::gamma_builder gb;
	gb.begin(predicate);

	/* add entry variables */
	auto ds = static_cast<const agg::branch_demand_set*>(dm.at(&node).get());
	std::unordered_map<const variable*, std::shared_ptr<jive::entryvar>> evmap;
	for (const auto & v : ds->cases_top) {
		JLM_DEBUG_ASSERT(svmap.last().find(v) != svmap.last().end());
		evmap[v] = gb.add_entryvar(svmap.last()[v]);
	}

	/* convert branch cases */
	std::unordered_map<const variable*, std::vector<jive::oport*>> xvmap;
	JLM_DEBUG_ASSERT(gb.nsubregions() == node.nchildren());
	for (size_t n = 0; n < gb.nsubregions(); n++) {
		auto & vmap = svmap.push_vmap();
		for (const auto & pair : evmap)
			vmap[pair.first] = pair.second->argument(n);

		convert_node(*node.child(n), dm, gb.region(n), svmap);

		for (const auto & v : ds->cases_bottom) {
			JLM_DEBUG_ASSERT(vmap.find(v) != vmap.end());
			xvmap[v].push_back(vmap[v]);
		}
		svmap.pop_vmap();
	}

	/* add exit variables */
	for (const auto & v : ds->cases_bottom) {
		JLM_DEBUG_ASSERT(xvmap.find(v) != xvmap.end());
		svmap.last()[v] = gb.add_exitvar(xvmap[v])->output();
	}

	gb.end();
	convert_basic_block(branch->join(), region, svmap);
}

static void
convert_loop_node(
	const agg::node & node,
	const agg::demand_map & dm,
	jive::region * region,
	scoped_vmap & svmap)
{
	JIVE_DEBUG_ASSERT(is_loop_structure(node.structure()));

	jive::theta_builder tb;
	tb.begin(region);

	auto & pvmap = svmap.last();
	auto & vmap = svmap.push_vmap();

	/* add loop variables */
	auto ds = dm.at(&node).get();
	std::unordered_map<const variable*, std::shared_ptr<jive::loopvar>> lvmap;
	for (const auto & v : ds->top) {
		JLM_DEBUG_ASSERT(pvmap.find(v) != pvmap.end());
		lvmap[v] = tb.add_loopvar(pvmap[v]);
		vmap[v] = lvmap[v]->value();
	}

	/* convert loop body */
	JLM_DEBUG_ASSERT(node.nchildren() == 1);
	convert_node(*node.child(0), dm, region, svmap);

	/* update loop variables */
	for (const auto & v : ds->top) {
		JLM_DEBUG_ASSERT(vmap.find(v) != vmap.end());
		JLM_DEBUG_ASSERT(lvmap.find(v) != lvmap.end());
		lvmap[v]->set_value(vmap[v]);
	}

	/* find predicate */
	auto lb = node.child(0);
	while (lb->nchildren() != 0)
		lb = lb->child(lb->nchildren()-1);
	JLM_DEBUG_ASSERT(is_block_structure(lb->structure()));
	auto bb = static_cast<const agg::block*>(&lb->structure())->basic_block();
	auto predicate = bb.last()->output(0);

	/* update variable map */
	JLM_DEBUG_ASSERT(vmap.find(predicate) != vmap.end());
	tb.end(vmap[predicate]);
	svmap.pop_vmap();
	for (const auto & v : ds->bottom) {
		JLM_DEBUG_ASSERT(pvmap.find(v) != pvmap.end());
		pvmap[v] = lvmap[v]->value();
	}
}

static void
convert_node(
	const agg::node & node,
	const agg::demand_map & dm,
	jive::region * region,
	scoped_vmap & svmap)
{
	static std::unordered_map<
		std::type_index,
		std::function<void(const agg::node&, const agg::demand_map&, jive::region*, scoped_vmap&)>
	> map ({
	  {std::type_index(typeid(agg::entry)), convert_entry_node}
	, {std::type_index(typeid(agg::exit)), convert_exit_node}
	, {std::type_index(typeid(agg::block)), convert_block_node}
	, {std::type_index(typeid(agg::linear)), convert_linear_node}
	, {std::type_index(typeid(agg::branch)), convert_branch_node}
	, {std::type_index(typeid(agg::loop)), convert_loop_node}
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(node.structure()))) != map.end());
	return map[std::type_index(typeid(node.structure()))](node, dm, region, svmap);
}

static jive::oport *
convert_cfg(
	const jlm::clg_node * function,
	jive::region * region,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(svmap.nvmaps() > 0);
	auto cfg = function->cfg();

	destruct_ssa(*cfg);
	restructure(cfg);
	auto root = agg::aggregate(*cfg);
	auto dm = agg::annotate(*root);

	auto & pvmap = svmap.last();
	svmap.push_vmap();
	auto & vmap = svmap.last();

	jive::lambda_builder lb;
	lb.begin(region, function->type());

	JLM_DEBUG_ASSERT(cfg->entry().narguments() == lb.region()->narguments());
	for (size_t n = 0; n < cfg->entry().narguments(); n++)
		vmap[cfg->entry().argument(n)] = lb.region()->argument(n);

	for (const auto & v : dm[root.get()].get()->top) {
		JLM_DEBUG_ASSERT(pvmap.find(v) != pvmap.end());
		vmap[v] = lb.add_dependency(pvmap[v]);
	}

	convert_node(*root, dm, lb.region(), svmap);

	/* FIXME: finding the exit node can be simplified once we have agg::node reductions */
	auto exit = root.get();
	while (!is_exit_structure(exit->structure())) {
		if (exit->nchildren() != 0)
			exit = exit->child(exit->nchildren()-1);
		else
			exit = nullptr;
	}
	JLM_DEBUG_ASSERT(exit != nullptr);
	auto xa = static_cast<const agg::exit*>(&exit->structure())->attribute();

	std::vector<jive::oport*> results;
	for (size_t n = 0; n < xa.nresults(); n++) {
		JLM_DEBUG_ASSERT(svmap.last().find(xa.result(n)) != svmap.last().end());
		results.push_back(svmap.last()[xa.result(n)]);
	}

	auto lambda = lb.end(results);
	svmap.pop_vmap();

	return lambda->output(0);
}

static jive::oport *
construct_lambda(
	const clg_node * function,
	jive::region * region,
	scoped_vmap & svmap)
{
	if (function->cfg() == nullptr)
		return region->graph()->import(function->type(), function->name());

	return convert_cfg(function, region, svmap);
}


static void
handle_scc(
	const std::unordered_set<const jlm::clg_node*> & scc,
	jive::graph * graph,
	scoped_vmap & svmap)
{
	auto & module = svmap.module();

	if (scc.size() == 1 && !(*scc.begin())->is_selfrecursive()) {
		const jlm::clg_node * function = *scc.begin();
		auto lambda = construct_lambda(function, graph->root(), svmap);
		JLM_DEBUG_ASSERT(svmap.module().variable(function));
		svmap.last()[svmap.module().variable(function)] = lambda;
		if (function->exported())
			graph->export_port(lambda, function->name());
	} else {
		jive::phi_builder pb;
		pb.begin(graph->root());
		svmap.push_vmap();

		/* FIXME: external dependencies */
		std::vector<std::shared_ptr<jive::recvar>> recvars;
		for (const auto & f : scc) {
			auto rv = pb.add_recvar(f->type());
			JLM_DEBUG_ASSERT(!module.variable(f));
			svmap.last()[module.variable(f)] = rv->value();
			recvars.push_back(rv);
		}

		size_t n = 0;
		for (const auto & f : scc) {
			auto lambda = construct_lambda(f, pb.region(), svmap);
			recvars[n++]->set_value(lambda);
		}

		svmap.pop_vmap();
		pb.end();

		for (const auto & f : scc) {
			auto value = recvars[n++]->value();
			svmap.last()[module.variable(f)] = value;
			if (f->exported())
				graph->export_port(value, f->name());
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
	scoped_vmap svmap(m);
	auto rvsdg = std::make_unique<jive::graph>();

/*
	dstrct::variable_map globals = convert_global_variables(m, graph);
*/
	auto sccs = m.clg().find_sccs();
	for (const auto & scc : sccs)
		handle_scc(scc, rvsdg.get(), svmap);

	return std::move(rvsdg);
}

}
