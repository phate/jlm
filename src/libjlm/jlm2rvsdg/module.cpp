/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/jlm2rvsdg/module.hpp>
#include <jlm/jlm/jlm2rvsdg/restructuring.hpp>

#include <jlm/jlm/ir/aggregation.hpp>
#include <jlm/jlm/ir/annotation.hpp>
#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/cfg-structure.hpp>
#include <jlm/jlm/ir/ipgraph.hpp>
#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/operators.hpp>
#include <jlm/jlm/ir/rvsdg.hpp>
#include <jlm/jlm/ir/ssa.hpp>
#include <jlm/jlm/ir/tac.hpp>

#include <jive/arch/address.h>
#include <jive/arch/addresstype.h>
#include <jive/arch/dataobject.h>
#include <jive/arch/memlayout-simple.h>
#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/float.h>
#include <jive/types/function.h>
#include <jive/rvsdg/binary.h>
#include <jive/rvsdg/control.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/phi.h>
#include <jive/rvsdg/region.h>
#include <jive/rvsdg/theta.h>
#include <jive/rvsdg/type.h>

#include <cmath>
#include <stack>

static inline jive::output *
create_undef_value(jive::region * region, const jive::type & type)
{
	jlm::undef_constant_op op(*static_cast<const jive::valuetype*>(&type));
	return jive::simple_node::create_normalized(region, op, {})[0];
}

namespace jlm {

typedef std::unordered_map<const variable*, jive::output*> vmap;

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
	JLM_DEBUG_ASSERT(is<assignment_op>(tac.operation()));
	vmap[tac.input(0)] = vmap[tac.input(1)];
}

static void
convert_select(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	JLM_DEBUG_ASSERT(is<select_op>(tac.operation()));
	JLM_DEBUG_ASSERT(tac.ninputs() == 3 && tac.noutputs() == 1);

	auto op = jive::match_op(1, {{1, 1}}, 0, 2);
	auto predicate = jive::simple_node::create_normalized(region, op, {vmap[tac.input(0)]})[0];

	auto gamma = jive::gamma_node::create(predicate, 2);
	auto ev1 = gamma->add_entryvar(vmap[tac.input(2)]);
	auto ev2 = gamma->add_entryvar(vmap[tac.input(1)]);
	auto ex = gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});
	vmap[tac.output(0)] = ex;
}

static void
convert_branch(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	JLM_DEBUG_ASSERT(is<branch_op>(tac.operation()));
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

	std::vector<jive::output*> operands;
	for (size_t n = 0; n < tac.ninputs(); n++) {
		JLM_DEBUG_ASSERT(vmap.find(tac.input(n)) != vmap.end());
		operands.push_back(vmap[tac.input(n)]);
	}

	auto results = jive::simple_node::create_normalized(region, static_cast<const jive::simple_op&>(
		tac.operation()), operands);

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
	const aggnode & node,
	const demandmap & dm,
	const jlm::function_node & function,
	lambda_builder & lb,
	scoped_vmap & svmap);

static jive::node *
convert_entry_node(
	const aggnode & node,
	const demandmap & dm,
	const jlm::function_node & function,
	lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is<entryaggnode>(&node));
	auto & entry = static_cast<const entryaggnode*>(&node)->attribute();
	auto ds = dm.at(&node).get();

	auto arguments = lb.begin_lambda(svmap.region(), {function.fcttype(), function.name()});
	svmap.push_scope(lb.subregion());

	auto & pvmap = svmap.vmap(svmap.nscopes()-2);
	auto & vmap = svmap.vmap();

	/* add arguments */
	JLM_DEBUG_ASSERT(entry.narguments() == arguments.size());
	for (size_t n = 0; n < entry.narguments(); n++)
		vmap[entry.argument(n)] = arguments[n];

	/* add dependencies and undefined values */
	for (const auto & v : ds->top) {
		if (pvmap.find(v) != pvmap.end()) {
			vmap[v] = lb.add_dependency(pvmap[v]);
		} else {
			auto value = create_undef_value(lb.subregion(), v->type());
			JLM_DEBUG_ASSERT(value);
			vmap[v] = value;
		}
	}

	return nullptr;
}

static jive::node *
convert_exit_node(
	const aggnode & node,
	const demandmap & dm,
	const jlm::function_node & function,
	lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is<exitaggnode>(&node));
	auto & xa = static_cast<const exitaggnode*>(&node)->attribute();

	std::vector<jive::output*> results;
	for (size_t n = 0; n < xa.nresults(); n++) {
		JLM_DEBUG_ASSERT(svmap.vmap().find(xa.result(n)) != svmap.vmap().end());
		results.push_back(svmap.vmap()[xa.result(n)]);
	}

	svmap.pop_scope();
	return lb.end_lambda(results);
}

static jive::node *
convert_block_node(
	const aggnode & node,
	const demandmap & dm,
	const jlm::function_node & function,
	lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is<blockaggnode>(&node));
	auto & bb = static_cast<const blockaggnode*>(&node)->basic_block();
	convert_basic_block(bb, svmap.region(), svmap.vmap());
	return nullptr;
}

static jive::node *
convert_linear_node(
	const aggnode & node,
	const demandmap & dm,
	const jlm::function_node & function,
	lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is<linearaggnode>(&node));

	jive::node * n = nullptr;
	for (const auto & child : node)
		n = convert_node(child, dm, function, lb, svmap);

	return n;
}

static jive::node *
convert_branch_node(
	const aggnode & node,
	const demandmap & dm,
	const jlm::function_node & function,
	lambda_builder & lb,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(is<branchaggnode>(&node));
	JLM_DEBUG_ASSERT(is<linearaggnode>(node.parent()));
	JLM_DEBUG_ASSERT(node.parent()->nchildren() == 2 && node.parent()->child(1) == &node);

	auto split = node.parent()->child(0);
	while (!is<blockaggnode>(split))
		split = split->child(split->nchildren()-1);
	auto & sb = dynamic_cast<const blockaggnode*>(split)->basic_block();

	JLM_DEBUG_ASSERT(is<branch_op>(sb.last()->operation()));
	auto predicate = svmap.vmap()[sb.last()->input(0)];
	auto gamma = jive::gamma_node::create(predicate, node.nchildren());

	/* add entry variables */
	auto & ds = dm.at(&node);
	std::unordered_map<const variable*, jive::gamma_input*> evmap;
	for (const auto & v : ds->top) {
		JLM_DEBUG_ASSERT(svmap.vmap().find(v) != svmap.vmap().end());
		evmap[v] = gamma->add_entryvar(svmap.vmap()[v]);
	}

	/* convert branch cases */
	std::unordered_map<const variable*, std::vector<jive::output*>> xvmap;
	JLM_DEBUG_ASSERT(gamma->nsubregions() == node.nchildren());
	for (size_t n = 0; n < gamma->nsubregions(); n++) {
		svmap.push_scope(gamma->subregion(n));
		for (const auto & pair : evmap)
			svmap.vmap()[pair.first] = pair.second->argument(n);

		convert_node(*node.child(n), dm, function, lb, svmap);

		for (const auto & v : ds->bottom) {
			JLM_DEBUG_ASSERT(svmap.vmap().find(v) != svmap.vmap().end());
			xvmap[v].push_back(svmap.vmap()[v]);
		}
		svmap.pop_scope();
	}

	/* add exit variables */
	for (const auto & v : ds->bottom) {
		JLM_DEBUG_ASSERT(xvmap.find(v) != xvmap.end());
		svmap.vmap()[v] = gamma->add_exitvar(xvmap[v]);
	}

	return nullptr;
}

static jive::node *
convert_loop_node(
	const aggnode & node,
	const demandmap & dm,
	const jlm::function_node & function,
	lambda_builder & lb,
	scoped_vmap & svmap)
{
	JIVE_DEBUG_ASSERT(is<loopaggnode>(&node));
	auto parent = svmap.region();

	auto theta = jive::theta_node::create(parent);

	svmap.push_scope(theta->subregion());
	auto & vmap = svmap.vmap();
	auto & pvmap = svmap.vmap(svmap.nscopes()-2);

	/* add loop variables */
	auto ds = dm.at(&node).get();
	JLM_DEBUG_ASSERT(ds->top == ds->bottom);
	std::unordered_map<const variable*, jive::theta_output*> lvmap;
	for (const auto & v : ds->top) {
		jive::output * value = nullptr;
		if (pvmap.find(v) == pvmap.end()) {
			value = create_undef_value(parent, v->type());
			JLM_DEBUG_ASSERT(value);
			pvmap[v] = value;
		} else {
			value = pvmap[v];
		}
		lvmap[v] = theta->add_loopvar(value);
		vmap[v] = lvmap[v]->argument();
	}

	/* convert loop body */
	JLM_DEBUG_ASSERT(node.nchildren() == 1);
	convert_node(*node.child(0), dm, function, lb, svmap);

	/* update loop variables */
	for (const auto & v : ds->top) {
		JLM_DEBUG_ASSERT(vmap.find(v) != vmap.end());
		JLM_DEBUG_ASSERT(lvmap.find(v) != lvmap.end());
		lvmap[v]->result()->divert_to(vmap[v]);
	}

	/* find predicate */
	auto lblock = node.child(0);
	while (lblock->nchildren() != 0)
		lblock = lblock->child(lblock->nchildren()-1);
	JLM_DEBUG_ASSERT(is<blockaggnode>(lblock));
	auto & bb = static_cast<const blockaggnode*>(lblock)->basic_block();
	JLM_DEBUG_ASSERT(is<branch_op>(bb.last()->operation()));
	auto predicate = bb.last()->input(0);

	/* update variable map */
	JLM_DEBUG_ASSERT(vmap.find(predicate) != vmap.end());
	theta->set_predicate(vmap[predicate]);
	svmap.pop_scope();
	for (const auto & v : ds->bottom) {
		JLM_DEBUG_ASSERT(pvmap.find(v) != pvmap.end());
		pvmap[v] = lvmap[v];
	}

	return nullptr;
}

static jive::node *
convert_node(
	const aggnode & node,
	const demandmap & dm,
	const jlm::function_node & function,
	lambda_builder & lb,
	scoped_vmap & svmap)
{
	static std::unordered_map<
		std::type_index,
		std::function<jive::node*(
			const aggnode&,
			const demandmap&,
			const jlm::function_node&,
			lambda_builder&,
			scoped_vmap&)
		>
	> map ({
	  {typeid(entryaggnode), convert_entry_node}, {typeid(exitaggnode), convert_exit_node}
	, {typeid(blockaggnode), convert_block_node}, {typeid(linearaggnode), convert_linear_node}
	, {typeid(branchaggnode), convert_branch_node}, {typeid(loopaggnode), convert_loop_node}
	});

	JLM_DEBUG_ASSERT(map.find(typeid(node)) != map.end());
	return map[typeid(node)](node, dm, function, lb, svmap);
}

static jive::output *
convert_cfg(
	const jlm::function_node & function,
	jive::region * region,
	scoped_vmap & svmap)
{
	auto cfg = function.cfg();

	destruct_ssa(*cfg);
	straighten(*cfg);
	purge(*cfg);

	restructure(cfg);
	auto root = aggregate(*cfg);
	auto dm = annotate(*root);

	lambda_builder lb;
	auto lambda = convert_node(*root, dm, function, lb, svmap);
	return lambda->output(0);
}

static jive::output *
construct_lambda(
	const ipgraph_node * node,
	jive::region * region,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const function_node*>(node));
	auto & function = *static_cast<const function_node*>(node);

	if (function.cfg() == nullptr)
		return region->graph()->add_import(function.type(), function.name());

	return convert_cfg(function, region, svmap);
}

static jive::output *
convert_tacs(const tacsvector_t & tacs, jive::region * region, scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(!tacs.empty());

	auto & vmap = svmap.vmap();
	for (const auto & tac : tacs)
		convert_tac(*tac, region, vmap);

	auto & tac = tacs.back();
	JLM_DEBUG_ASSERT(tac->noutputs() == 1);
	return vmap[tac->output(0)];
}

static jive::output *
convert_data_node(
	const jlm::ipgraph_node * node,
	jive::region * region,
	scoped_vmap & svmap)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const data_node*>(node));
	auto n = static_cast<const data_node*>(node);
	auto & m = svmap.module();

	jive::output * data = nullptr;
	if (n->initialization().empty()) {
		data = region->graph()->add_import(n->type(), n->name());
	} else {
		jlm::delta_builder db;
		auto r = db.begin(region, n->linkage(), n->constant());
		auto & pv = svmap.vmap();
		svmap.push_scope(r);

		/* add dependencies */
		for (const auto & dp : *node) {
			auto v = m.variable(dp);
			JLM_DEBUG_ASSERT(pv.find(v) != pv.end());
			auto argument = db.add_dependency(pv[v]);
			svmap.vmap()[v] = argument;
		}

		data = db.end(convert_tacs(n->initialization(), r, svmap));
		svmap.pop_scope();
	}

	return data;
}

static void
handle_scc(
	const std::unordered_set<const jlm::ipgraph_node*> & scc,
	jive::graph * graph,
	scoped_vmap & svmap)
{
	auto & m = svmap.module();

	static std::unordered_map<
		std::type_index,
		std::function<jive::output*(const ipgraph_node*, jive::region*, scoped_vmap&)>
	> map({
	  {typeid(data_node), convert_data_node}
	, {typeid(function_node), construct_lambda}
	});

	if (scc.size() == 1 && !(*scc.begin())->is_selfrecursive()) {
		auto & node = *scc.begin();
		JLM_DEBUG_ASSERT(map.find(typeid(*node)) != map.end());
		auto output = map[typeid(*node)](node, graph->root(), svmap);

		auto v = m.variable(node);
		JLM_DEBUG_ASSERT(v);
		svmap.vmap()[v] = output;
		if (v->exported())
			graph->add_export(output, v->name());
	} else {
		jive::phi_builder pb;
		pb.begin_phi(graph->root());
		svmap.push_scope(pb.region());

		auto & pvmap = svmap.vmap(svmap.nscopes()-2);
		auto & vmap = svmap.vmap();

		/* add recursion variables */
		std::unordered_map<const variable*, std::shared_ptr<jive::recvar>> recvars;
		for (const auto & node : scc) {
			auto rv = pb.add_recvar(node->type());
			auto v = m.variable(node);
			JLM_DEBUG_ASSERT(v);
			vmap[v] = rv->value();
			JLM_DEBUG_ASSERT(recvars.find(v) == recvars.end());
			recvars[v] = rv;
		}

		/* add dependencies */
		for (const auto & node : scc) {
			for (const auto & dep : *node) {
				auto v = m.variable(dep);
				JLM_DEBUG_ASSERT(v);
				if (recvars.find(v) == recvars.end())
					vmap[v] = pb.add_dependency(pvmap[v]);
			}
		}

		/* convert SCC nodes */
		for (const auto & node : scc) {
			auto output = map[typeid(*node)](node, pb.region(), svmap);
			recvars[m.variable(node)]->set_value(output);
		}

		svmap.pop_scope();
		pb.end_phi();

		/* add phi outputs */
		for (const auto & node : scc) {
			auto v = m.variable(node);
			auto value = recvars[v]->value();
			svmap.vmap()[v] = value;
			if (v->exported())
				graph->add_export(value, v->name());
		}
	}
}

std::unique_ptr<jlm::rvsdg>
construct_rvsdg(const module & m)
{
	auto rvsdg = std::make_unique<jlm::rvsdg>(m.target_triple(), m.data_layout());
	auto graph = rvsdg->graph();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	/* FIXME: we currently cannot handle flattened_binary_op in jlm2llvm pass */
	jive::binary_op::normal_form(graph)->set_flatten(false);

	scoped_vmap svmap(m, graph->root());

	/* convert ipgraph nodes */
	auto sccs = m.ipgraph().find_sccs();
	for (const auto & scc : sccs)
		handle_scc(scc, graph, svmap);

	return std::move(rvsdg);
}

}
