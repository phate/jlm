/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/operators/gamma.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/opt/invariance.hpp>
#include <jlm/util/stats.hpp>
#include <jlm/util/strfmt.hpp>
#include <jlm/util/time.hpp>

#include <jive/rvsdg/gamma.hpp>
#include <jive/rvsdg/theta.hpp>
#include <jive/rvsdg/traverser.hpp>

namespace jlm {

class invstat final : public stat {
public:
	virtual
	~invstat()
	{}

	invstat()
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
		return strfmt("INV ",
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
invariance(jive::region * region);

static void
gamma_invariance(jive::structural_node * node)
{
	JLM_DEBUG_ASSERT(jive::is<jive::gamma_op>(node));
	auto gamma = static_cast<jive::gamma_node*>(node);

	for (size_t n = 0; n < gamma->noutputs(); n++) {
		auto output = static_cast<jive::gamma_output*>(gamma->output(n));
		if (auto no = is_invariant(output))
			output->divert_users(no);
	}
}

static void
theta_invariance(jive::structural_node * node)
{
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(node));
	auto theta = static_cast<jive::theta_node*>(node);

	/* FIXME: In order to also redirect state variables,
		we need to know whether a loop terminates.*/

	for (const auto & lv : *theta) {
		if (jive::is_invariant(lv) && !jive::is<loopstatetype>(lv->argument()->type()))
			lv->divert_users(lv->input()->origin());
	}
}

static void
invariance(jive::region * region)
{
	for (auto node : jive::topdown_traverser(region)) {
		if (jive::is<jive::simple_op>(node))
			continue;

		JLM_DEBUG_ASSERT(jive::is<jive::structural_op>(node));
		auto strnode = static_cast<jive::structural_node*>(node);
		for (size_t n = 0; n < strnode->nsubregions(); n++)
			invariance(strnode->subregion(n));

		if (jive::is<jive::gamma_op>(node)) {
			gamma_invariance(strnode);
			continue;
		}

		if (jive::is<jive::theta_op>(node)) {
			theta_invariance(strnode);
			continue;
		}
	}
}

static void
invariance(rvsdg_module & rm, const stats_descriptor & sd)
{
	invstat stat;

	stat.start(*rm.graph());
	invariance(rm.graph()->root());
	stat.end(*rm.graph());

	if (sd.print_inv_stat)
		sd.print_stat(stat);
}

/* ivr class */

ivr::~ivr()
{}

void
ivr::run(rvsdg_module & module, const stats_descriptor & sd)
{
	invariance(module, sd);
}

}
