/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm2rvsdg/module.hpp>
#include <jlm/jlm2rvsdg/restructuring.hpp>

#include <jlm/ir/aggregation/aggregation.hpp>
#include <jlm/ir/aggregation/annotation.hpp>
#include <jlm/ir/aggregation/node.hpp>
#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/clg.hpp>
#include <jlm/ir/data.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/ssa.hpp>
#include <jlm/ir/tac.hpp>

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

static inline jive::oport *
create_undef_value(jive::region * region, const jive::base::type & type)
{
	if (auto t = dynamic_cast<const jive::bits::type*>(&type))
		return jive_bitconstant_undefined(region, t->nbits());

	return nullptr;
}

namespace jlm {

typedef std::unordered_map<const variable*, jive::oport*> vmap;

class scoped_vmap final {
public:
	inline
	~scoped_vmap()
	{
		pop_scope();
		JLM_DEBUG_ASSERT(nscopes() == 0);
	}

	inline
	scoped_vmap(const jlm::module & module, jive::region * region)
	: module_(module)
	{
		push_scope(region);
	}

	inline size_t
	nscopes() const noexcept
	{
		JLM_DEBUG_ASSERT(vmaps_.size() == regions_.size());
		return vmaps_.size();
	}

	inline jlm::vmap &
	vmap(size_t n) noexcept
	{
		JLM_DEBUG_ASSERT(n < nscopes());
		return *vmaps_[n];
	}

	inline jlm::vmap &
	vmap() noexcept
	{
		JLM_DEBUG_ASSERT(nscopes() > 0);
		return vmap(nscopes()-1);
	}

	inline jive::region *
	region(size_t n) noexcept
	{
		JLM_DEBUG_ASSERT(n < nscopes());
		return regions_[n];
	}

	inline jive::region *
	region() noexcept
	{
		JLM_DEBUG_ASSERT(nscopes() > 0);
		return region(nscopes()-1);
	}

	inline void
	push_scope(jive::region * region)
	{
		vmaps_.push_back(std::make_unique<jlm::vmap>());
		regions_.push_back(region);
	}

	inline void
	pop_scope()
	{
		vmaps_.pop_back();
		regions_.pop_back();
	}

	const jlm::module &
	module() const noexcept
	{
		return module_;
	}

private:
	const jlm::module & module_;
	std::vector<std::unique_ptr<jlm::vmap>> vmaps_;
	std::vector<jive::region*> regions_;
};

static void
convert_assignment(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	JLM_DEBUG_ASSERT(is_assignment_op(tac.operation()));
	JLM_DEBUG_ASSERT(tac.ninputs() == 1 && tac.noutputs() == 1);
	vmap[tac.output(0)] = vmap[tac.input(0)];
}

static void
convert_select(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	JLM_DEBUG_ASSERT(is_select_op(tac.operation()));
	JLM_DEBUG_ASSERT(tac.ninputs() == 3 && tac.noutputs() == 1);

	auto op = jive::match_op(1, {{1, 0}}, 1, 2);
	auto predicate = jive::create_normalized(region, op, {vmap[tac.input(0)]})[0];

	jive::gamma_builder gb;
	gb.begin(predicate);
	auto ev1 = gb.add_entryvar(vmap[tac.input(1)]);
	auto ev2 = gb.add_entryvar(vmap[tac.input(2)]);
	auto ex = gb.add_exitvar({ev1->argument(0), ev2->argument(1)});
	vmap[tac.output(0)] = ex->output();
	gb.end();
}

static void
convert_branch(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	JLM_DEBUG_ASSERT(is_branch_op(tac.operation()));
}

static void
convert_tac(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	static std::unordered_map<
		std::type_index,
		std::function<void(const jlm::tac&, jive::region*, jlm::vmap&)>
	> map({
	  {std::type_index(typeid(assignment_op)), convert_assignment}
	, {std::type_index(typeid(select_op)), convert_select}
	, {std::type_index(typeid(branch_op)), convert_branch}
	});

	if (map.find(std::type_index(typeid(tac.operation()))) != map.end())
		return map[std::type_index(typeid(tac.operation()))](tac, region, vmap);

	std::vector<jive::oport*> operands;
	for (size_t n = 0; n < tac.ninputs(); n++) {
		JLM_DEBUG_ASSERT(vmap.find(tac.input(n)) != vmap.end());
		operands.push_back(vmap[tac.input(n)]);
	}

	auto results = jive::create_normalized(region, tac.operation(), operands);

	JLM_DEBUG_ASSERT(results.size() == tac.noutputs());
	for (size_t n = 0; n < tac.noutputs(); n++)
		vmap[tac.output(n)] = results[n];
}

static void
convert_basic_block(const basic_block & bb, jive::region * region, jlm::vmap & vmap)
{
	for (const auto & tac: bb)
		convert_tac(*tac, region, vmap);
}

static jive::node *
convert_node(
	const agg::node & node,
	const agg::demand_map & dm,
	const jlm::clg_node & function,
	jive::lambda_builder & lb,
	scoped_vmap & svmap);

static jive::node *
convert_entry_node(
	const agg::node & node,
	const agg::demand_map & dm,
	const jlm::clg_node & function,
	jive::lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_entry_structure(node.structure()));
	auto entry = static_cast<const agg::entry*>(&node.structure())->attribute();
	auto ds = dm.at(&node).get();

	lb.begin(svmap.region(), function.type());
	svmap.push_scope(lb.region());

	auto & pvmap = svmap.vmap(svmap.nscopes()-2);
	auto & vmap = svmap.vmap();

	/* add arguments */
	JLM_DEBUG_ASSERT(entry.narguments() == lb.region()->narguments());
	for (size_t n = 0; n < entry.narguments(); n++)
		vmap[entry.argument(n)] = lb.region()->argument(n);

	/* add dependencies and undefined values */
	for (const auto & v : ds->top) {
		if (pvmap.find(v) != pvmap.end()) {
			vmap[v] = lb.add_dependency(pvmap[v]);
		} else {
			auto value = create_undef_value(lb.region(), v->type());
			JLM_DEBUG_ASSERT(value);
			vmap[v] = value;
		}
	}

	return nullptr;
}

static jive::node *
convert_exit_node(
	const agg::node & node,
	const agg::demand_map & dm,
	const jlm::clg_node & function,
	jive::lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_exit_structure(node.structure()));
	auto xa = static_cast<const agg::exit*>(&node.structure())->attribute();

	std::vector<jive::oport*> results;
	for (size_t n = 0; n < xa.nresults(); n++) {
		JLM_DEBUG_ASSERT(svmap.vmap().find(xa.result(n)) != svmap.vmap().end());
		results.push_back(svmap.vmap()[xa.result(n)]);
	}

	svmap.pop_scope();
	return lb.end(results);
}

static jive::node *
convert_block_node(
	const agg::node & node,
	const agg::demand_map & dm,
	const jlm::clg_node & function,
	jive::lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_block_structure(node.structure()));
	auto & bb = static_cast<const agg::block*>(&node.structure())->basic_block();
	convert_basic_block(bb, svmap.region(), svmap.vmap());
	return nullptr;
}

static jive::node *
convert_linear_node(
	const agg::node & node,
	const agg::demand_map & dm,
	const jlm::clg_node & function,
	jive::lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_linear_structure(node.structure()));

	jive::node * n = nullptr;
	for (const auto & child : node)
		n = convert_node(child, dm, function, lb, svmap);

	return n;
}

static jive::node *
convert_branch_node(
	const agg::node & node,
	const agg::demand_map & dm,
	const jlm::clg_node & function,
	jive::lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is_branch_structure(node.structure()));

	convert_node(*node.child(0), dm, function, lb, svmap);

	auto split = node.child(0);
	while (!is_block_structure(split->structure()))
		split = split->child(split->nchildren()-1);
	auto & sb = dynamic_cast<const agg::block*>(&split->structure())->basic_block();

	JLM_DEBUG_ASSERT(is_branch_op(sb.last()->operation()));
	auto predicate = svmap.vmap()[sb.last()->input(0)];
	jive::gamma_builder gb;
	gb.begin(predicate);

	/* add entry variables */
	auto ds = static_cast<const agg::branch_demand_set*>(dm.at(&node).get());
	std::unordered_map<const variable*, std::shared_ptr<jive::entryvar>> evmap;
	for (const auto & v : ds->cases_top) {
		JLM_DEBUG_ASSERT(svmap.vmap().find(v) != svmap.vmap().end());
		evmap[v] = gb.add_entryvar(svmap.vmap()[v]);
	}

	/* convert branch cases */
	std::unordered_map<const variable*, std::vector<jive::oport*>> xvmap;
	JLM_DEBUG_ASSERT(gb.nsubregions() == node.nchildren()-1);
	for (size_t n = 0; n < gb.nsubregions(); n++) {
		svmap.push_scope(gb.region(n));
		for (const auto & pair : evmap)
			svmap.vmap()[pair.first] = pair.second->argument(n);

		convert_node(*node.child(n+1), dm, function, lb, svmap);

		for (const auto & v : ds->cases_bottom) {
			JLM_DEBUG_ASSERT(svmap.vmap().find(v) != svmap.vmap().end());
			xvmap[v].push_back(svmap.vmap()[v]);
		}
		svmap.pop_scope();
	}

	/* add exit variables */
	for (const auto & v : ds->cases_bottom) {
		JLM_DEBUG_ASSERT(xvmap.find(v) != xvmap.end());
		svmap.vmap()[v] = gb.add_exitvar(xvmap[v])->output();
	}

	gb.end();
	return nullptr;
}

static jive::node *
convert_loop_node(
	const agg::node & node,
	const agg::demand_map & dm,
	const jlm::clg_node & function,
	jive::lambda_builder & lb,
	scoped_vmap & svmap)
{
	JIVE_DEBUG_ASSERT(is_loop_structure(node.structure()));
	auto parent = svmap.region();

	jive::theta_builder tb;
	tb.begin(parent);

	svmap.push_scope(tb.region());
	auto & vmap = svmap.vmap();
	auto & pvmap = svmap.vmap(svmap.nscopes()-2);

	/* add loop variables */
	auto ds = dm.at(&node).get();
	JLM_DEBUG_ASSERT(ds->top == ds->bottom);
	std::unordered_map<const variable*, std::shared_ptr<jive::loopvar>> lvmap;
	for (const auto & v : ds->top) {
		jive::oport * value = nullptr;
		if (pvmap.find(v) == pvmap.end()) {
			value = create_undef_value(parent, v->type());
			JLM_DEBUG_ASSERT(value);
			pvmap[v] = value;
		} else {
			value = pvmap[v];
		}
		lvmap[v] = tb.add_loopvar(value);
		vmap[v] = lvmap[v]->value();
	}

	/* convert loop body */
	JLM_DEBUG_ASSERT(node.nchildren() == 1);
	convert_node(*node.child(0), dm, function, lb, svmap);

	/* update loop variables */
	for (const auto & v : ds->top) {
		JLM_DEBUG_ASSERT(vmap.find(v) != vmap.end());
		JLM_DEBUG_ASSERT(lvmap.find(v) != lvmap.end());
		lvmap[v]->set_value(vmap[v]);
	}

	/* find predicate */
	auto lblock = node.child(0);
	while (lblock->nchildren() != 0)
		lblock = lblock->child(lblock->nchildren()-1);
	JLM_DEBUG_ASSERT(is_block_structure(lblock->structure()));
	auto & bb = static_cast<const agg::block*>(&lblock->structure())->basic_block();
	JLM_DEBUG_ASSERT(is_branch_op(bb.last()->operation()));
	auto predicate = bb.last()->input(0);

	/* update variable map */
	JLM_DEBUG_ASSERT(vmap.find(predicate) != vmap.end());
	tb.end(vmap[predicate]);
	svmap.pop_scope();
	for (const auto & v : ds->bottom) {
		JLM_DEBUG_ASSERT(pvmap.find(v) != pvmap.end());
		pvmap[v] = lvmap[v]->value();
	}

	return nullptr;
}

static jive::node *
convert_node(
	const agg::node & node,
	const agg::demand_map & dm,
	const jlm::clg_node & function,
	jive::lambda_builder & lb,
	scoped_vmap & svmap)
{
	static std::unordered_map<
		std::type_index,
		std::function<jive::node*(
			const agg::node&,
			const agg::demand_map&,
			const jlm::clg_node&,
			jive::lambda_builder&,
			scoped_vmap&)
		>
	> map ({
	  {std::type_index(typeid(agg::entry)), convert_entry_node}
	, {std::type_index(typeid(agg::exit)), convert_exit_node}
	, {std::type_index(typeid(agg::block)), convert_block_node}
	, {std::type_index(typeid(agg::linear)), convert_linear_node}
	, {std::type_index(typeid(agg::branch)), convert_branch_node}
	, {std::type_index(typeid(agg::loop)), convert_loop_node}
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(node.structure()))) != map.end());
	return map[std::type_index(typeid(node.structure()))](node, dm, function, lb, svmap);
}

static jive::oport *
convert_cfg(
	const jlm::clg_node & function,
	jive::region * region,
	scoped_vmap & svmap)
{
	auto cfg = function.cfg();

	destruct_ssa(*cfg);
	straighten(*cfg);
	purge(*cfg);

	restructure(cfg);
	auto root = agg::aggregate(*cfg);
	auto dm = agg::annotate(*root);

	jive::lambda_builder lb;
	auto lambda = convert_node(*root, dm, function, lb, svmap);
	return lambda->output(0);
}

static jive::oport *
construct_lambda(
	const clg_node & function,
	jive::region * region,
	scoped_vmap & svmap)
{
	if (function.cfg() == nullptr)
		return region->graph()->import(function.type(), function.name());

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
		auto lambda = construct_lambda(*function, graph->root(), svmap);
		JLM_DEBUG_ASSERT(svmap.module().variable(function));
		svmap.vmap()[svmap.module().variable(function)] = lambda;
		if (function->exported())
			graph->export_port(lambda, function->name());
	} else {
		jive::phi_builder pb;
		pb.begin(graph->root());
		svmap.push_scope(pb.region());

		/* FIXME: external dependencies */
		std::vector<std::shared_ptr<jive::recvar>> recvars;
		for (const auto & f : scc) {
			auto rv = pb.add_recvar(f->type());
			JLM_DEBUG_ASSERT(module.variable(f));
			svmap.vmap()[module.variable(f)] = rv->value();
			recvars.push_back(rv);
		}

		size_t n = 0;
		for (const auto & f : scc) {
			auto lambda = construct_lambda(*f, pb.region(), svmap);
			recvars[n++]->set_value(lambda);
		}

		svmap.pop_scope();
		pb.end();

		n = 0;
		for (const auto & f : scc) {
			auto value = recvars[n++]->value();
			svmap.vmap()[module.variable(f)] = value;
			if (f->exported())
				graph->export_port(value, f->name());
		}
	}
}

static inline jive::oport *
convert_expression(const expr & e, jive::region * region)
{
	std::vector<jive::oport*> operands;
	for (size_t n = 0; n < e.noperands(); n++)
		operands.push_back(convert_expression(e.operand(n), region));

	return jive::create_normalized(region, e.operation(), operands)[0];
}

static inline void
convert_globals(jive::graph * rvsdg, scoped_vmap & svmap)
{
	auto & m = svmap.module();

	for (const auto & gv : m) {
		auto variable = gv.first;
		auto & expression = *gv.second;

		jlm::data_builder db;
		auto region = db.begin(rvsdg->root());
		auto data = db.end(convert_expression(expression, region));

		svmap.vmap()[variable] = data;
		if (variable->exported())
			rvsdg->export_port(data, gv.first->name());
	}
}

std::unique_ptr<jive::graph>
construct_rvsdg(const module & m)
{
	auto rvsdg = std::make_unique<jive::graph>();
	scoped_vmap svmap(m, rvsdg->root());

	convert_globals(rvsdg.get(), svmap);

	/* convert functions */
	auto sccs = m.clg().find_sccs();
	for (const auto & scc : sccs)
		handle_scc(scc, rvsdg.get(), svmap);

	return std::move(rvsdg);
}

}
