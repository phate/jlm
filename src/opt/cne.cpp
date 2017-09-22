/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/data.hpp>
#include <jlm/opt/cne.hpp>

#include <jive/types/function/fctlambda.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/phi.h>
#include <jive/vsdg/simple_node.h>
#include <jive/vsdg/theta.h>
#include <jive/vsdg/traverser.h>

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

	inline void
	set_processed(congruence_set * set)
	{
		processed_.insert(set);
	}

	inline bool
	processed(congruence_set * set) const noexcept
	{
		return processed_.find(set) != processed_.end();
	}

private:
	std::unordered_set<congruence_set*> processed_;
	std::unordered_set<std::unique_ptr<congruence_set>> sets_;
	std::unordered_map<const jive::output*, congruence_set*> outputs_;
};

/* mark phase */

static bool
congruent_results(
	const jive::structural_output * o1,
	const jive::structural_output * o2,
	const cnectx & ctx)
{
	JLM_DEBUG_ASSERT(o1->node() && o1->node() == o2->node());

	auto r1 = o1->results.first;
	auto r2 = o2->results.first;
	while (r1 != nullptr) {
		if (!ctx.congruent(r1, r2))
			return false;
		r1 = r1->output_result_list.next;
		r2 = r2->output_result_list.next;
	}
	JLM_DEBUG_ASSERT(r2 == nullptr);

	return true;
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
		ctx.mark(a1, a2);
		a1 = a1->input_argument_list.next;
		a2 = a2->input_argument_list.next;
	}
	JLM_DEBUG_ASSERT(a2 == nullptr);
}

static void
mark_arguments(const jive::structural_node * node, cnectx & ctx)
{
	for (size_t i1 = 0; i1 < node->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < node->ninputs(); i2++) {
			if (ctx.congruent(node->input(i1), node->input(i2)))
				mark_arguments(node->input(i1), node->input(i2), ctx);
		}
	}
}

static void
mark(jive::region*, cnectx&);

static void
mark_gamma(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_gamma_op(node->operation()));

	/* mark entry variables */
	mark_arguments(node, ctx);

	for (size_t n = 0; n < node->nsubregions(); n++)
		mark(node->subregion(n), ctx);

	/* mark exit variables */
	for (size_t o1 = 0; o1 < node->noutputs(); o1++) {
		for (size_t o2 = o1+1; o2 < node->noutputs(); o2++) {
			if (congruent_results(node->output(o1), node->output(o2), ctx))
				ctx.mark(node->output(o1), node->output(o2));
		}
	}
}

static void
mark_theta(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(jive::is_theta_op(node->operation()));

	/* mark loop entries */
	mark_arguments(node, ctx);

	mark(node->subregion(0), ctx);

	/* mark loop exits */
	for (size_t o1 = 0; o1 < node->noutputs(); o1++) {
		for (size_t o2 = o1+1; o2 < node->noutputs(); o2++) {
			if (ctx.congruent(node->input(o1), node->input(o2))
			&& congruent_results(node->output(o1), node->output(o2), ctx))
				ctx.mark(node->output(o1), node->output(o2));
		}
	}
}

static void
mark_lambda(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(jive::fct::is_lambda_op(node->operation()));
	mark_arguments(node, ctx);
	mark(node->subregion(0), ctx);
}

static void
mark_phi(const jive::structural_node * node, cnectx & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::phi_op*>(&node->operation()));
	mark_arguments(node, ctx);
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
	if (ctx.processed(set))
		return;

	for (auto & other : *set)
		other->replace(output);

	ctx.set_processed(set);
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
	auto root = graph.root();

	cnectx ctx;
	mark(root, ctx);
	divert(root, ctx);
}

}
