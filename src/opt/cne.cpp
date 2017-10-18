/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/data.hpp>
#include <jlm/opt/cne.hpp>
#include <jlm/util/stats.hpp>

#include <jive/types/function/fctlambda.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/phi.h>
#include <jive/vsdg/simple_node.h>
#include <jive/vsdg/theta.h>
#include <jive/vsdg/traverser.h>

#if defined(CNEMARKTIME) || defined(CNEDIVERTTIME)
	#include <iostream>
#endif

namespace jlm {

typedef std::unordered_set<jive::output*> congruence_set;

class cnectx {
public:
	inline void
	mark(jive::output * o1, jive::output * o2)
	{
		auto s1 = set(o1);
		auto s2 = set(o2);

		if (s1 == s2)
			return;

		if (s2->size() < s1->size()) {
			s1 = outputs_[o2];
			s2 = outputs_[o1];
		}

		for (auto & o : *s1) {
			s2->insert(o);
			outputs_[o] = s2;
		}
	}

	inline void
	mark(const jive::node * n1, const jive::node * n2)
	{
		JLM_DEBUG_ASSERT(n1->noutputs() == n2->noutputs());

		for (size_t n = 0; n < n1->noutputs(); n++)
			mark(n1->output(n), n2->output(n));
	}

	inline bool
	congruent(jive::output * o1, jive::output * o2) const noexcept
	{
		if (o1 == o2)
			return true;

		auto it = outputs_.find(o1);
		if (it == outputs_.end())
			return false;

		return it->second->find(o2) != it->second->end();
	}

	inline bool
	congruent(const jive::input * i1, const jive::input * i2) const noexcept
	{
		return congruent(i1->origin(), i2->origin());
	}

	congruence_set *
	set(jive::output * output) noexcept
	{
		if (outputs_.find(output) == outputs_.end()) {
			std::unique_ptr<congruence_set> set(new congruence_set({output}));
			outputs_[output] = set.get();
			sets_.insert(std::move(set));
		}

		return outputs_[output];
	}

private:
	std::unordered_set<std::unique_ptr<congruence_set>> sets_;
	std::unordered_map<const jive::output*, congruence_set*> outputs_;
};

class vset {
public:
	void
	insert(const jive::output * o1, const jive::output * o2)
	{
		auto it = sets_.find(o1);
		if (it != sets_.end())
			sets_[o1].insert(o2);
		else
			sets_[o1] = {o2};

		it = sets_.find(o2);
		if (it != sets_.end())
			sets_[o2].insert(o1);
		else
			sets_[o2] = {o1};
	}

	bool
	visited(const jive::output * o1, const jive::output * o2) const
	{
		auto it = sets_.find(o1);
		if (it == sets_.end())
			return false;

		return it->second.find(o2) != it->second.end();
	}

private:
	std::unordered_map<
		const jive::output*,
		std::unordered_set<const jive::output*>
	> sets_;
};

static bool
is_theta_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument && is_theta_node(argument->region()->node());
}

static bool
is_gamma_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument && is_gamma_node(argument->region()->node());
}

static bool
is_simple_node(const jive::node * node)
{
	return jive::is_opnode<jive::simple_op>(node);
}

/* mark phase */

static bool
congruent(
	jive::output * o1,
	jive::output * o2,
	vset & vs,
	cnectx & ctx)
{
	if (ctx.congruent(o1, o2) || vs.visited(o1, o2))
		return true;

	if (o1->type() != o2->type())
		return false;

	auto a1 = static_cast<jive::argument*>(o1);
	auto a2 = static_cast<jive::argument*>(o2);
	auto so1 = static_cast<jive::structural_output*>(o1);
	auto so2 = static_cast<jive::structural_output*>(o2);
	if (is_theta_argument(o1) && is_theta_argument(o2)) {
		JLM_DEBUG_ASSERT(o1->region()->node() == o2->region()->node());
		vs.insert(a1, a2);
		auto i1 = a1->input(), i2 = a2->input();
		if (!congruent(a1->input()->origin(), a2->input()->origin(), vs, ctx))
			return false;

		auto output1 = o1->region()->node()->output(i1->index());
		auto output2 = o2->region()->node()->output(i2->index());
		return congruent(output1, output2, vs, ctx);
	}

	if (is_theta_node(o1->node()) && is_theta_node(o2->node()) && o1->node() == o2->node()) {
		vs.insert(o1, o2);
		auto r1 = so1->results.first;
		auto r2 = so2->results.first;
		return congruent(r1->origin(), r2->origin(), vs, ctx);
	}

	if (is_gamma_node(o1->node()) && is_gamma_node(o2->node()) && o1->node() == o2->node()) {
		auto r1 = so1->results.first, r2 = so2->results.first;
		while (r1 != nullptr) {
			JLM_DEBUG_ASSERT(r1->region() == r2->region());
			if (!congruent(r1->origin(), r2->origin(), vs, ctx))
				return false;
			r1 = r1->output_result_list.next;
			r2 = r2->output_result_list.next;
		}
		JLM_DEBUG_ASSERT(r2 == nullptr);
		return true;
	}

	if (is_gamma_argument(o1) && is_gamma_argument(o2)) {
		JLM_DEBUG_ASSERT(o1->region()->node() == o2->region()->node());
		return congruent(a1->input()->origin(), a2->input()->origin(), vs, ctx);
	}

	if (is_simple_node(o1->node()) && is_simple_node(o2->node())
	&& o1->node()->operation() == o2->node()->operation()
	&& o1->node()->ninputs() == o2->node()->ninputs()
	&& o1->index() == o2->index()) {
		auto n1 = o1->node(), n2 = o2->node();
		for (size_t n = 0; n < n1->ninputs(); n++) {
			auto origin1 = n1->input(n)->origin();
			auto origin2 = n2->input(n)->origin();
			if (!congruent(origin1, origin2, vs, ctx))
				return false;
		}
		return true;
	}

	return false;
}

static bool
congruent(jive::output * o1, jive::output * o2, cnectx & ctx)
{
	vset vs;
	return congruent(o1, o2, vs, ctx);
}

static void
mark_arguments(
	const jive::structural_input * i1,
	const jive::structural_input * i2,
	cnectx & ctx)
{
	JLM_DEBUG_ASSERT(i1->node() && i1->node() == i2->node());

	auto a1 = i1->arguments.first;
	auto a2 = i2->arguments.first;
	while (a1 != nullptr) {
		if (congruent(a1, a2, ctx))
			ctx.mark(a1, a2);
		a1 = a1->input_argument_list.next;
		a2 = a2->input_argument_list.next;
	}
	JLM_DEBUG_ASSERT(a2 == nullptr);
}

static void
mark(jive::region*, cnectx&);

static void
mark_gamma(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_gamma_op(node->operation()));

	/* mark entry variables */
	for (size_t i1 = 1; i1 < node->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < node->ninputs(); i2++)
			mark_arguments(node->input(i1), node->input(i2), ctx);
	}

	for (size_t n = 0; n < node->nsubregions(); n++)
		mark(node->subregion(n), ctx);

	/* mark exit variables */
	for (size_t o1 = 0; o1 < node->noutputs(); o1++) {
		for (size_t o2 = o1+1; o2 < node->noutputs(); o2++) {
			if (congruent(node->output(o1), node->output(o2), ctx))
				ctx.mark(node->output(o1), node->output(o2));
		}
	}
}

static void
mark_theta(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_theta_node(node));

	/* mark loop variables */
	for (size_t i1 = 0; i1 < node->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < node->ninputs(); i2++) {
			auto a1 = node->input(i1)->arguments.first;
			auto a2 = node->input(i2)->arguments.first;
			while (a1 != nullptr) {
				if (congruent(a1, a2, ctx)) {
					ctx.mark(a1, a2);
					ctx.mark(node->output(i1), node->output(i2));
				}
				a1 = a1->input_argument_list.next;
				a2 = a2->input_argument_list.next;
			}
			JLM_DEBUG_ASSERT(a2 == nullptr);
		}
	}

	mark(node->subregion(0), ctx);
}

static void
mark_lambda(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(jive::fct::is_lambda_op(node->operation()));

	/* mark dependencies */
	for (size_t i1 = 0; i1 < node->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < node->ninputs(); i2++) {
			auto input1 = node->input(i1);
			auto input2 = node->input(i2);
			if (ctx.congruent(input1, input2))
				ctx.mark(input1->arguments.first, input2->arguments.first);
		}
	}

	mark(node->subregion(0), ctx);
}

static void
mark_phi(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::phi_op*>(&node->operation()));

	/* mark dependencies */
	for (size_t i1 = 0; i1 < node->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < node->ninputs(); i2++) {
			auto input1 = node->input(i1);
			auto input2 = node->input(i2);
			if (ctx.congruent(input1, input2))
				ctx.mark(input1->arguments.first, input2->arguments.first);
		}
	}

	mark(node->subregion(0), ctx);
}

static void
mark_data(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_data_op(node->operation()));
}

static void
mark(const jive::structural_node * node, cnectx & ctx)
{
	static std::unordered_map<
		std::type_index
	, void(*)(const jive::structural_node*, cnectx&)
	> map({
	  {std::type_index(typeid(jive::gamma_op)), mark_gamma}
	, {std::type_index(typeid(jive::theta_op)), mark_theta}
	, {std::type_index(typeid(jive::fct::lambda_op)), mark_lambda}
	, {std::type_index(typeid(jive::phi_op)), mark_phi}
	, {std::type_index(typeid(data_op)), mark_data}
	});

	std::type_index index(typeid(node->operation()));
	JLM_DEBUG_ASSERT(map.find(index) != map.end());
	map[index](node, ctx);
}

static void
mark(const jive::simple_node * node, cnectx & ctx)
{
	if (node->ninputs() == 0) {
		jive::node * other;
		JIVE_LIST_ITERATE(node->region()->top_nodes, other, region_top_node_list) {
			if (other != node && node->operation() == other->operation()) {
				ctx.mark(node, other);
				break;
			}
		}
		return;
	}

	auto set = ctx.set(node->input(0)->origin());
	for (const auto & origin : *set) {
		for (const auto & user : *origin) {
			auto other = user->node();
			if (!other
			|| other == node
			|| other->operation() != node->operation()
			|| other->ninputs() != node->ninputs())
				continue;

			size_t n;
			for (n = 0; n < node->ninputs(); n++) {
				if (!ctx.congruent(node->input(n), other->input(n)))
					break;
			}
			if (n == node->ninputs())
				ctx.mark(node, other);
		}
	}
}

static void
mark(jive::region * region, cnectx & ctx)
{
	for (const auto & node : jive::topdown_traverser(region)) {
		if (auto simple = dynamic_cast<const jive::simple_node*>(node))
			mark(simple, ctx);
		else
			mark(static_cast<const jive::structural_node*>(node), ctx);
	}
}

/* divert phase */

static void
divert_users(jive::output * output, cnectx & ctx)
{
	auto set = ctx.set(output);
	for (auto & other : *set)
		other->replace(output);
	set->clear();
}

static void
divert_outputs(jive::node * node, cnectx & ctx)
{
	for (size_t n = 0; n < node->noutputs(); n++)
		divert_users(node->output(n), ctx);
}

static void
divert_arguments(jive::region * region, cnectx & ctx)
{
	for (size_t n = 0; n < region->narguments(); n++)
		divert_users(region->argument(n), ctx);
}

static void
divert(jive::region*, cnectx&);

static void
divert_gamma(jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_gamma_node(node));

	jive::gamma gamma(node);
	for (auto ev = gamma.begin_entryvar(); ev != gamma.end_entryvar(); ev++) {
		for (size_t n = 0; n < ev->narguments(); n++)
			divert_users(ev->argument(n), ctx);
	}

	for (size_t r = 0; r < node->nsubregions(); r++)
		divert(node->subregion(r), ctx);

	divert_outputs(node, ctx);
}

static void
divert_theta(jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_theta_node(node));
	auto subregion = node->subregion(0);

	jive::theta theta(node);
	for (const auto & lv : theta) {
		JLM_DEBUG_ASSERT(ctx.set(lv.argument())->size() == ctx.set(lv.output())->size());
		divert_users(lv.argument(), ctx);
		divert_users(lv.output(), ctx);
	}

	divert(subregion, ctx);
}

static void
divert_lambda(jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(jive::fct::is_lambda_op(node->operation()));

	divert_arguments(node->subregion(0), ctx);
	divert(node->subregion(0), ctx);
}

static void
divert_phi(jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::phi_op*>(&node->operation()));

	divert_arguments(node->subregion(0), ctx);
	divert(node->subregion(0), ctx);
}

static void
divert_data(jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_data_op(node->operation()));
}

static void
divert(jive::structural_node * node, cnectx & ctx)
{
	static std::unordered_map<
		std::type_index,
		void(*)(jive::structural_node*, cnectx&)
	> map({
	  {std::type_index(typeid(jive::gamma_op)), divert_gamma}
	, {std::type_index(typeid(jive::theta_op)), divert_theta}
	, {std::type_index(typeid(jive::fct::lambda_op)), divert_lambda}
	, {std::type_index(typeid(jive::phi_op)), divert_phi}
	, {std::type_index(typeid(data_op)), divert_data}
	});

	std::type_index index(typeid(node->operation()));
	JLM_DEBUG_ASSERT(map.find(index) != map.end());
	map[index](node, ctx);
}

static void
divert(jive::region * region, cnectx & ctx)
{
	for (const auto & node : jive::topdown_traverser(region)) {
		if (auto simple = dynamic_cast<jive::simple_node*>(node))
			divert_outputs(simple, ctx);
		else
			divert(static_cast<jive::structural_node*>(node), ctx);
	}
}

void
cne(jive::graph & graph)
{
	cnectx ctx;

	auto mark_ = [&](jive::graph & graph)
	{
		mark(graph.root(), ctx);
	};

	auto divert_ = [&](jive::graph & graph)
	{
		divert(graph.root(), ctx);
	};

	statscollector mc, dc;
	mc.run(mark_, graph);
	dc.run(divert_, graph);

#ifdef CNEMARKTIME
	std::cout << "CNEMARKTIME: "
	          << mc.nnodes_before() << " "
	          << mc.ninputs_before() << " "
	          << mc.time() << "\n";
#endif

#ifdef CNEDIVERTTIME
	std::cout << "CNEDIVERTTIME: "
	          << dc.nnodes_before() << " "
	          << dc.ninputs_before() << " "
	          << dc.time() << "\n";
#endif
}

}
