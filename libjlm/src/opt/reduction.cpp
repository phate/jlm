/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/binary.hpp>
#include <jive/rvsdg/gamma.hpp>
#include <jive/rvsdg/statemux.hpp>

#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/reduction.hpp>
#include <jlm/util/stats.hpp>
#include <jlm/util/time.hpp>

namespace jlm {

class redstat final : public stat {
public:
	virtual
	~redstat()
	{}

	redstat()
	: nnodes_before_(0), nnodes_after_(0)
	, ninputs_before_(0), ninputs_after_(0)
	{}

	void
	start(const jive::graph & graph) noexcept
	{
		nnodes_before_ = jive::nnodes(graph.root());
		ninputs_before_ = jive::ninputs(graph.root());
		timer_.start();
	}

	void
	end(const jive::graph & graph) noexcept
	{
		nnodes_after_ = jive::nnodes(graph.root());
		ninputs_after_ = jive::ninputs(graph.root());
		timer_.stop();
	}

	virtual std::string
	to_str() const override
	{
		return strfmt("RED ",
			nnodes_before_, " ", nnodes_after_, " ",
			ninputs_before_, " ", ninputs_after_, " ",
			timer_.ns()
		);
	}

private:
	size_t nnodes_before_, nnodes_after_;
	size_t ninputs_before_, ninputs_after_;
	jlm::timer timer_;
};

static void
enable_mux_reductions(jive::graph & graph)
{
	auto nf = graph.node_normal_form(typeid(jive::mux_op));
	auto mnf = static_cast<jive::mux_normal_form*>(nf);
	mnf->set_mutable(true);
	mnf->set_mux_mux_reducible(true);
	mnf->set_multiple_origin_reducible(true);
}

static void
enable_store_reductions(jive::graph & graph)
{
	auto nf = jlm::store_op::normal_form(&graph);
	nf->set_mutable(true);
	nf->set_store_mux_reducible(true);
	nf->set_store_store_reducible(true);
	nf->set_store_alloca_reducible(true);
	nf->set_multiple_origin_reducible(true);
}

static void
enable_load_reductions(jive::graph & graph)
{
	auto nf = jlm::load_op::normal_form(&graph);
	nf->set_mutable(true);
	nf->set_load_mux_reducible(true);
	nf->set_load_store_reducible(true);
	nf->set_load_alloca_reducible(true);
	nf->set_multiple_origin_reducible(true);
	nf->set_load_store_state_reducible(true);
	nf->set_load_store_alloca_reducible(true);
	nf->set_load_load_state_reducible(true);
}

static void
enable_gamma_reductions(jive::graph & graph)
{
	auto nf = jive::gamma_op::normal_form(&graph);
	nf->set_mutable(true);
	nf->set_predicate_reduction(true);
	nf->set_control_constant_reduction(true);
}

static void
enable_unary_reductions(jive::graph & graph)
{
	auto nf = jive::unary_op::normal_form(&graph);
	nf->set_mutable(true);
	nf->set_reducible(true);
}

static void
enable_binary_reductions(jive::graph & graph)
{
	auto nf = jive::binary_op::normal_form(&graph);
	nf->set_mutable(true);
	nf->set_reducible(true);
}

static void
reduce(rvsdg_module & rm, const stats_descriptor & sd)
{
	auto & graph = *rm.graph();

	redstat stat;
	stat.start(graph);

	enable_mux_reductions(graph);
	enable_store_reductions(graph);
	enable_load_reductions(graph);
	enable_gamma_reductions(graph);
	enable_unary_reductions(graph);
	enable_binary_reductions(graph);

	graph.normalize();
	stat.end(graph);

	if (sd.print_reduction_stat)
		sd.print_stat(stat);
}

/* nodereduction class */

nodereduction::~nodereduction()
{}

void
nodereduction::run(rvsdg_module & module, const stats_descriptor & sd)
{
	reduce(module, sd);
}

}
