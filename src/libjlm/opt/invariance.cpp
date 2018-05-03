/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/opt/invariance.hpp>

#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/theta.h>
#include <jive/rvsdg/traverser.h>

#ifdef INVTIME
#include <chrono>
#include <iostream>
#endif

namespace jlm {

static bool
is_invariant(const jive::gamma_output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output->result(0)->origin());
	if (!argument)
		return false;

	size_t n;
	auto origin = argument->input()->origin();
	for (n = 1; n < output->nresults(); n++) {
		auto argument = dynamic_cast<const jive::argument*>(output->result(n)->origin());
		if (argument == nullptr || argument->input()->origin() != origin)
			break;
	}

	return n == output->nresults();
}

static void
invariance(jive::region * region);

static void
gamma_invariance(jive::structural_node * node)
{
	JLM_DEBUG_ASSERT(jive::is<jive::gamma_op>(node));
	auto gamma = static_cast<jive::gamma_node*>(node);

	for (size_t n = 0; n < gamma->noutputs(); n++) {
		auto output = static_cast<jive::gamma_output*>(gamma->output(n));
		if (is_invariant(output)) {
			auto no = static_cast<jive::argument*>(output->result(0)->origin())->input()->origin();
			output->divert_users(no);
		}
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
		if (jive::is_invariant(lv)
		&& dynamic_cast<const jive::valuetype*>(&lv->argument()->type()))
			lv->divert_users(lv->input()->origin());
	}
}

static void
invariance(jive::region * region)
{
	for (auto node : jive::topdown_traverser(region)) {
		if (dynamic_cast<const jive::simple_op*>(&node->operation()))
			continue;

		JLM_DEBUG_ASSERT(dynamic_cast<const jive::structural_node*>(node));
		auto strnode = static_cast<jive::structural_node*>(node);
		for (size_t n = 0; n < strnode->nsubregions(); n++)
			invariance(strnode->subregion(n));

		if (jive::is<jive::gamma_op>(node->operation())) {
			gamma_invariance(strnode);
			continue;
		}

		if (jive::is<jive::theta_op>(node->operation())) {
			theta_invariance(strnode);
			continue;
		}
	}
}

void
invariance(jive::graph & graph)
{
	auto root = graph.root();

	#ifdef INVTIME
		auto nnodes = jive::nnodes(root);
		auto start = std::chrono::high_resolution_clock::now();
	#endif

	invariance(root);

	#ifdef INVTIME
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "INVTIME: "
		          << nnodes
		          << " " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()
		          << "\n";
	#endif
}

}
