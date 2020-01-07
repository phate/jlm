/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/dne.hpp>
#include <jlm/util/stats.hpp>
#include <jlm/util/time.hpp>

#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/phi.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/structural-node.h>
#include <jive/rvsdg/theta.h>
#include <jive/rvsdg/traverser.h>

namespace jlm {

class dnestat final : public stat {
public:
	virtual
	~dnestat()
	{}

	dnestat()
	: nnodes_before_(0), nnodes_after_(0)
	, ninputs_before_(0), ninputs_after_(0)
	{}

	void
	start_mark_stat(const jive::graph & graph) noexcept
	{
		nnodes_before_ = jive::nnodes(graph.root());
		ninputs_before_ = jive::ninputs(graph.root());
		marktimer_.start();
	}

	void
	end_mark_stat() noexcept
	{
		marktimer_.stop();
	}

	void
	start_sweep_stat() noexcept
	{
		sweeptimer_.start();
	}

	void
	end_sweep_stat(const jive::graph & graph) noexcept
	{
		nnodes_after_ = jive::nnodes(graph.root());
		ninputs_after_ = jive::ninputs(graph.root());
		sweeptimer_.stop();
	}

	virtual std::string
	to_str() const override
	{
		return strfmt("DNE ",
			nnodes_before_, " ", nnodes_after_, " ",
			ninputs_before_, " ", ninputs_after_, " ",
			marktimer_.ns(), " ", sweeptimer_.ns()
		);
	}

private:
	size_t nnodes_before_, nnodes_after_;
	size_t ninputs_before_, ninputs_after_;
	jlm::timer marktimer_, sweeptimer_;
};


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
	using namespace jive;

	auto a = dynamic_cast<const argument*>(output);
	return a && is<gamma_op>(a->region()->node());
}

static bool
is_theta_argument(const jive::output * output)
{
	using namespace jive;

	auto a = dynamic_cast<const argument*>(output);
	return a && is<theta_op>(a->region()->node());
}

static bool
is_lambda_output(const jive::output * output)
{
	return output->node()
	    && dynamic_cast<const lambda_op*>(&output->node()->operation());
}

static bool
is_lambda_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument
	    && argument->region()->node()
	    && dynamic_cast<const lambda_op*>(&argument->region()->node()->operation());
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

	if (jive::is<jive::gamma_op>(output->node())) {
		auto gamma = static_cast<const jive::gamma_node*>(output->node());
		auto soutput = static_cast<const jive::structural_output*>(output);
		mark(gamma->predicate()->origin(), ctx);
		for (const auto & result : soutput->results)
			mark(result.origin(), ctx);
		return;
	}

	if (is_gamma_argument(output)) {
		auto argument = static_cast<const jive::argument*>(output);
		mark(argument->input()->origin(), ctx);
		return;
	}

	if (dynamic_cast<const jive::theta_output*>(output)) {
		auto lv = static_cast<const jive::theta_output*>(output);
		mark(lv->node()->predicate()->origin(), ctx);
		mark(lv->result()->origin(), ctx);
		mark(lv->input()->origin(), ctx);
		return;
	}

	if (is_theta_argument(output)) {
		auto theta = output->region()->node();
		auto argument = static_cast<const jive::argument*>(output);
		mark(theta->output(argument->input()->index()), ctx);
		mark(argument->input()->origin(), ctx);
		return;
	}

	if (is_lambda_output(output)) {
		auto soutput = static_cast<const jive::structural_output*>(output);
		for (size_t n = 0; n < soutput->node()->subregion(0)->nresults(); n++)
			mark(soutput->node()->subregion(0)->result(n)->origin(), ctx);
		return;
	}

	if (is_lambda_argument(output)) {
		auto argument = static_cast<const jive::argument*>(output);
		if (argument->input())
			mark(argument->input()->origin(), ctx);
		return;
	}

	if (is_phi_output(output)) {
		auto soutput = static_cast<const jive::structural_output*>(output);
		mark(soutput->results.first()->origin(), ctx);
		return;
	}

	if (is_phi_argument(output)) {
		auto argument = static_cast<const jive::argument*>(output);
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
sweep_delta(jive::structural_node * node, const dnectx & ctx)
{
	JLM_DEBUG_ASSERT(is<delta_op>(node));
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
	JLM_DEBUG_ASSERT(dynamic_cast<const lambda_op*>(&node->operation()));
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
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(node));
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
			JLM_DEBUG_ASSERT(node->output(n)->results.first() == nullptr);
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
	JLM_DEBUG_ASSERT(jive::is<jive::gamma_op>(node));

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

		bool alive = false;
		for (const auto & argument : input->arguments) {
			if (ctx.is_alive(&argument)) {
				alive = true;
				break;
			}
		}
		if (!alive) {
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
	, {std::type_index(typeid(jlm::lambda_op)), sweep_lambda}
	, {std::type_index(typeid(jive::phi_op)), sweep_phi}
	, {typeid(jlm::delta_op), sweep_delta}
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
	JLM_DEBUG_ASSERT(region->bottom_nodes.empty());
}

static void
mark(const jive::graph & graph, dnectx & ctx)
{
	for (size_t n = 0; n < graph.root()->nresults(); n++)
		mark(graph.root()->result(n)->origin(), ctx);
}

static void
sweep(jive::graph & graph, dnectx & ctx)
{
	sweep(graph.root(), ctx);
	for (ssize_t n = graph.root()->narguments()-1; n >= 0; n--) {
		if (!ctx.is_alive(graph.root()->argument(n)))
			graph.root()->remove_argument(n);
	}
}

void
dne(rvsdg_module & rm, const stats_descriptor & sd)
{
	auto & graph = *rm.graph();

	dnectx ctx;
	dnestat ds;

	ds.start_mark_stat(graph);
	mark(graph, ctx);
	ds.end_mark_stat();

	ds.start_sweep_stat();
	sweep(graph, ctx);
	ds.end_sweep_stat(graph);

	if (sd.print_dne_stat)
		sd.print_stat(ds);
}
}
