/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/rvsdg/statemux.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm {

class redstat final : public Statistics {
public:
	virtual
	~redstat()
	{}

	redstat()
	: Statistics(Statistics::Id::ReduceNodes)
  , nnodes_before_(0), nnodes_after_(0)
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
	ToString() const override
	{
		return strfmt("RED ",
			nnodes_before_, " ", nnodes_after_, " ",
			ninputs_before_, " ", ninputs_after_, " ",
			timer_.ns()
		);
	}

  static std::unique_ptr<redstat>
  Create()
  {
    return std::make_unique<redstat>();
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
	auto nf = StoreOperation::GetNormalForm(&graph);
	nf->set_mutable(true);
	nf->set_store_mux_reducible(true);
	nf->set_store_store_reducible(true);
	nf->set_store_alloca_reducible(true);
	nf->set_multiple_origin_reducible(true);
}

static void
enable_load_reductions(jive::graph & graph)
{
	auto nf = LoadOperation::GetNormalForm(&graph);
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
reduce(
  RvsdgModule & rm,
  StatisticsCollector & statisticsCollector)
{
	auto & graph = rm.Rvsdg();
  auto statistics = redstat::Create();

	statistics->start(graph);

	enable_mux_reductions(graph);
	enable_store_reductions(graph);
	enable_load_reductions(graph);
	enable_gamma_reductions(graph);
	enable_unary_reductions(graph);
	enable_binary_reductions(graph);

	graph.normalize();
	statistics->end(graph);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* nodereduction class */

nodereduction::~nodereduction()
{}

void
nodereduction::run(
  RvsdgModule & module,
  StatisticsCollector & statisticsCollector)
{
	reduce(module, statisticsCollector);
}

}
