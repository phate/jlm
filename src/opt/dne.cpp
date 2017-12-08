/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/data.hpp>
#include <jlm/opt/dne.hpp>
#include <jlm/util/stats.hpp>

#include <jive/types/function/fctlambda.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/phi.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/structural-node.h>
#include <jive/rvsdg/theta.h>
#include <jive/rvsdg/traverser.h>

#if defined(DNEMARKTIME) || defined(DNESWEEPTIME)
	#include <iostream>
#endif

namespace jlm {

class dnectx {
public:
	inline void
	mark(const jive::output * output)
	{
		outputs_.insert(output);
	}

	inline bool
	is_alive(const jive::output * output) const noexcept
	{
		return outputs_.find(output) != outputs_.end();
	}

	inline bool
	is_alive(const jive::node * node) const noexcept
	{
		for (size_t n = 0; n < node->noutputs(); n++) {
			if (is_alive(node->output(n)))
				return true;
		}

		return false;
	}

private:
	std::unordered_set<const jive::output*> outputs_;
};

static bool
is_import(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument && !argument->region()->node();
}

static bool
is_gamma_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument && is_gamma_node(argument->region()->node());
}

static bool
is_theta_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument && is_theta_node(argument->region()->node());
}

static bool
is_lambda_output(const jive::output * output)
{
	return output->node()
	    && dynamic_cast<const jive::fct::lambda_op*>(&output->node()->operation());
}

static bool
is_lambda_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument
	    && argument->region()->node()
	    && dynamic_cast<const jive::fct::lambda_op*>(&argument->region()->node()->operation());
}

static bool
is_phi_output(const jive::output * output)
{
	return output->node()
	    && dynamic_cast<const jive::phi_op*>(&output->node()->operation());
}

static bool
is_phi_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument
	    && argument->region()->node()
	    && dynamic_cast<const jive::phi_op*>(&argument->region()->node()->operation());
}

/* mark phase */

static void
mark(const jive::output*, dnectx&);

static void
mark(const jive::output * output, dnectx & ctx)
{
	if (ctx.is_alive(output))
		return;

	ctx.mark(output);

	if (is_import(output))
		return;

	auto argument = static_cast<const jive::argument*>(output);
	auto soutput = static_cast<const jive::structural_output*>(output);
	if (is_gamma_node(output->node())) {
		mark(output->node()->input(0)->origin(), ctx);
		jive::result * result;
		JIVE_LIST_ITERATE(soutput->results, result, output_result_list)
			mark(result->origin(), ctx);
		return;
	}

	if (is_gamma_argument(output)) {
		mark(argument->input()->origin(), ctx);
		return;
	}

	if (is_theta_node(output->node())) {
		mark(soutput->node()->subregion(0)->result(0)->origin(), ctx);
		mark(soutput->results.first->origin(), ctx);
		mark(output->node()->input(output->index())->origin(), ctx);
		return;
	}

	if (is_theta_argument(output)) {
		auto theta = output->region()->node();
		mark(theta->output(argument->input()->index()), ctx);
		mark(argument->input()->origin(), ctx);
		return;
	}

	if (is_lambda_output(output)) {
		for (size_t n = 0; n < soutput->node()->subregion(0)->nresults(); n++)
			mark(soutput->node()->subregion(0)->result(n)->origin(), ctx);
		return;
	}

	if (is_lambda_argument(output)) {
		if (argument->input())
			mark(argument->input()->origin(), ctx);
		return;
	}

	if (is_phi_output(output)) {
		mark(soutput->results.first->origin(), ctx);
		return;
	}

	if (is_phi_argument(output)) {
		if (argument->input()) mark(argument->input()->origin(), ctx);
		else mark(argument->region()->result(argument->index())->origin(), ctx);
		return;
	}

	for (size_t n = 0; n < output->node()->ninputs(); n++)
		mark(output->node()->input(n)->origin(), ctx);
}

/* sweep phase */

static void
sweep(jive::region * region, const dnectx & ctx);

static void
sweep_data(jive::structural_node * node, const dnectx & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const data_op*>(&node->operation()));
	JLM_DEBUG_ASSERT(node->noutputs() == 1);

	if (!ctx.is_alive(node)) {
		remove(node);
		return;
	}
}

static void
sweep_phi(jive::structural_node * node, const dnectx & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::phi_op*>(&node->operation()));
	auto subregion = node->subregion(0);

	if (!ctx.is_alive(node)) {
		remove(node);
		return;
	}

	/* remove outputs and results */
	for (ssize_t n = subregion->nresults()-1; n >= 1; n--) {
		auto result = subregion->result(n);
		if (!ctx.is_alive(result->output())
		&& !ctx.is_alive(subregion->argument(result->index()))) {
			subregion->remove_result(n);
			node->remove_output(n);
		}
	}

	sweep(subregion, ctx);

	/* remove dead arguments and dependencies */
	for (ssize_t n = subregion->narguments()-1; n >= 0; n--) {
		auto argument = subregion->argument(n);
		auto input = argument->input();
		if (!ctx.is_alive(argument)) {
			subregion->remove_argument(n);
			if (input) node->remove_input(input->index());
		}
	}
}

static void
sweep_lambda(jive::structural_node * node, const dnectx & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::fct::lambda_op*>(&node->operation()));
	auto subregion = node->subregion(0);

	if (!ctx.is_alive(node)) {
		remove(node);
		return;
	}

	sweep(subregion, ctx);

	/* remove inputs and arguments */
	for (ssize_t n = subregion->narguments()-1; n >= 0; n--) {
		auto argument = subregion->argument(n);
		if (argument->input() == nullptr)
			continue;

		if (!ctx.is_alive(argument)) {
			size_t index = argument->input()->index();
			subregion->remove_argument(n);
			node->remove_input(index);
		}
	}
}

static void
sweep_theta(jive::structural_node * node, const dnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_theta_node(node));
	auto subregion = node->subregion(0);

	if (!ctx.is_alive(node)) {
		remove(node);
		return;
	}

	/* remove results */
	for (ssize_t n = subregion->nresults()-1; n >= 1; n--) {
		if (!ctx.is_alive(subregion->argument(n-1)) && !ctx.is_alive(node->output(n-1)))
			subregion->remove_result(n);
	}

	sweep(subregion, ctx);

	/* remove outputs, inputs, and arguments */
	for (ssize_t n = subregion->narguments()-1; n >= 0; n--) {
		if (!ctx.is_alive(subregion->argument(n)) && !ctx.is_alive(node->output(n))) {
			JLM_DEBUG_ASSERT(node->output(n)->results.first == nullptr);
			subregion->remove_argument(n);
			node->remove_input(n);
			node->remove_output(n);
		}
	}

	JLM_DEBUG_ASSERT(node->ninputs() == node->noutputs());
	JLM_DEBUG_ASSERT(subregion->narguments() == subregion->nresults()-1);
}

static void
sweep_gamma(jive::structural_node * node, const dnectx & ctx)
{
	JLM_DEBUG_ASSERT(is_gamma_node(node));

	if (!ctx.is_alive(node)) {
		remove(node);
		return;
	}

	/* remove outputs and results */
	for (ssize_t n = node->noutputs()-1; n >= 0; n--) {
		if (ctx.is_alive(node->output(n)))
			continue;

		for (size_t r = 0; r < node->nsubregions(); r++)
			node->subregion(r)->remove_result(n);
		node->remove_output(n);
	}

	for (size_t r = 0; r < node->nsubregions(); r++)
		sweep(node->subregion(r), ctx);

	/* remove arguments and inputs */
	for (ssize_t n = node->ninputs()-1; n >=  1; n--) {
		auto input = node->input(n);

		jive::argument * argument;
		JIVE_LIST_ITERATE(input->arguments, argument, input_argument_list) {
			if (ctx.is_alive(argument))
				break;
		}
		if (argument == nullptr) {
			for (size_t r = 0; r < node->nsubregions(); r++)
				node->subregion(r)->remove_argument(n-1);
			node->remove_input(n);
		}
	}
}

static void
sweep(jive::structural_node * node, const dnectx & ctx)
{
	static std::unordered_map<
		std::type_index
	, void(*)(jive::structural_node*, const dnectx&)
	> map({
	  {std::type_index(typeid(jive::gamma_op)), sweep_gamma}
	, {std::type_index(typeid(jive::theta_op)), sweep_theta}
	, {std::type_index(typeid(jive::fct::lambda_op)), sweep_lambda}
	, {std::type_index(typeid(jive::phi_op)), sweep_phi}
	, {std::type_index(typeid(jlm::data_op)), sweep_data}
	});

	std::type_index index(typeid(node->operation()));
	JLM_DEBUG_ASSERT(map.find(index) != map.end());
	map[index](node, ctx);
}

static void
sweep(jive::simple_node * node, const dnectx & ctx)
{
	if (!ctx.is_alive(node))
		remove(node);
}

static void
sweep(jive::region * region, const dnectx & ctx)
{
	for (const auto & node : jive::bottomup_traverser(region)) {
		if (auto simple = dynamic_cast<jive::simple_node*>(node))
			sweep(simple, ctx);
		else
			sweep(static_cast<jive::structural_node*>(node), ctx);
	}
	JLM_DEBUG_ASSERT(region->bottom_nodes.first == nullptr);
}

void
dne(jive::graph & graph)
{
	dnectx ctx;

	auto mark_ = [&](jive::graph & graph)
	{
		for (size_t n = 0; n < graph.root()->nresults(); n++)
			mark(graph.root()->result(n)->origin(), ctx);
	};

	auto sweep_ = [&](jive::graph & graph)
	{
		sweep(graph.root(), ctx);
		for (ssize_t n = graph.root()->narguments()-1; n >= 0; n--) {
			if (!ctx.is_alive(graph.root()->argument(n)))
				graph.root()->remove_argument(n);
		}
	};

	statscollector mc, sc;
	mc.run(mark_, graph);
	sc.run(sweep_, graph);

#ifdef DNEMARKTIME
	std::cout << "DNEMARKTIME: "
	          << mc.nnodes_before() << " "
	          << mc.ninputs_before() << " "
	          << mc.time() << "\n";
#endif

#ifdef DNESWEEPTIME
	std::cout << "DNESWEEPTIME: "
	          << sc.nnodes_before() << " "
	          << sc.ninputs_before() << " "
	          << sc.time() << "\n";
#endif
}

}
