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

#include <jive/rvsdg/gamma.hpp>
#include <jive/rvsdg/phi.hpp>
#include <jive/rvsdg/simple-node.hpp>
#include <jive/rvsdg/structural-node.hpp>
#include <jive/rvsdg/theta.hpp>
#include <jive/rvsdg/traverser.hpp>

namespace jlm {

/* dnestat class */

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
is_phi_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument
	    && argument->region()->node()
	    && is<jive::phi::operation>(argument->region()->node());
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

	if (auto gammaOutput = is_gamma_output(output)) {
		mark(gammaOutput->node()->predicate()->origin(), ctx);
		for (const auto & result : gammaOutput->results)
			mark(result.origin(), ctx);
		return;
	}

	if (auto argument = is_gamma_argument(output)) {
		mark(argument->input()->origin(), ctx);
		return;
	}

	if (auto thetaOutput = is_theta_output(output)) {
		mark(thetaOutput->node()->predicate()->origin(), ctx);
		mark(thetaOutput->result()->origin(), ctx);
		mark(thetaOutput->input()->origin(), ctx);
		return;
	}

	if (auto thetaArgument = is_theta_argument(output)) {
		auto thetaInput = static_cast<const jive::theta_input*>(thetaArgument->input());
		mark(thetaInput->output(), ctx);
		mark(thetaInput->origin(), ctx);
		return;
	}

	if (auto o = dynamic_cast<const lambda::output*>(output)) {
		for (auto & result : o->node()->fctresults())
			mark(result.origin(), ctx);
		return;
	}

	if (is<lambda::fctargument>(output))
		return;

	if (auto cv = dynamic_cast<const lambda::cvargument*>(output)) {
		mark(cv->input()->origin(), ctx);
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

	auto node = jive::node_output::node(output);
	JLM_ASSERT(node);
	for (size_t n = 0; n < node->ninputs(); n++)
		mark(node->input(n)->origin(), ctx);
}

/* sweep phase */

static void
sweep(jive::region * region, const dnectx & ctx);

static void
sweep_delta(jive::structural_node * node, const dnectx & ctx)
{
	JLM_ASSERT(is<delta::operation>(node));
	JLM_ASSERT(node->noutputs() == 1);

	if (!ctx.is_alive(node)) {
		remove(node);
		return;
	}
}

static void
sweep_phi(jive::structural_node * node, const dnectx & ctx)
{
	JLM_ASSERT(is<jive::phi::operation>(node));
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
	JLM_ASSERT(is<lambda::operation>(node));
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
	JLM_ASSERT(jive::is<jive::theta_op>(node));
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
			JLM_ASSERT(node->output(n)->results.first() == nullptr);
			subregion->remove_argument(n);
			node->remove_input(n);
			node->remove_output(n);
		}
	}

	JLM_ASSERT(node->ninputs() == node->noutputs());
	JLM_ASSERT(subregion->narguments() == subregion->nresults()-1);
}

static void
sweep_gamma(jive::structural_node * node, const dnectx & ctx)
{
	JLM_ASSERT(jive::is<jive::gamma_op>(node));

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
	  {typeid(jive::gamma_op),       sweep_gamma}
	, {typeid(jive::theta_op),       sweep_theta}
	, {typeid(lambda::operation),    sweep_lambda}
	, {typeid(jive::phi::operation), sweep_phi}
	, {typeid(delta::operation),     sweep_delta}
	});

	auto & op = node->operation();
	JLM_ASSERT(map.find(typeid(op)) != map.end());
	map[typeid(op)](node, ctx);
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
	JLM_ASSERT(region->bottom_nodes.empty());
}

static void
mark(const jive::region & region, dnectx & ctx)
{
	for (size_t n = 0; n < region.nresults(); n++)
		mark(region.result(n)->origin(), ctx);
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

static void
dne(rvsdg_module & rm, const stats_descriptor & sd)
{
	auto & graph = *rm.graph();

	dnectx ctx;
	dnestat ds;

	ds.start_mark_stat(graph);
	mark(*graph.root(), ctx);
	ds.end_mark_stat();

	ds.start_sweep_stat();
	sweep(graph, ctx);
	ds.end_sweep_stat(graph);

	if (sd.print_dne_stat)
		sd.print_stat(ds);
}

/* dne class */

dne::~dne()
{}

void
dne::run(jive::region & region)
{
	dnectx ctx;
	mark(region, ctx);
	sweep(&region, ctx);
}

void
dne::run(rvsdg_module & module, const stats_descriptor & sd)
{
	jlm::dne(module, sd);
}

}
