/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/opt/invariance.hpp>

#include <jive/vsdg/gamma.h>
#include <jive/vsdg/theta.h>
#include <jive/vsdg/traverser.h>

#ifdef INVTIME
#include <chrono>
#include <iostream>
#endif

namespace jlm {

static void
invariance(jive::region * region);

static void
gamma_invariance(jive::structural_node * node)
{
	JLM_DEBUG_ASSERT(is_gamma_op(node->operation()));

	jive::gamma gamma(node);
	for (auto it = gamma.begin_exitvar(); it != gamma.end_exitvar(); it++) {
		auto argument = dynamic_cast<jive::argument*>(it->result(0)->origin());
		if (argument == nullptr)
			continue;

		size_t n;
		auto input = argument->input();
		for (n = 1; n < it->nresults(); n++) {
			auto argument = dynamic_cast<jive::argument*>(it->result(n)->origin());
			if (argument == nullptr || argument->input() != input)
				break;
		}

		if (n == it->nresults())
			it->output()->replace(input->origin());
	}
}

static void
theta_invariance(jive::structural_node * node)
{
	JLM_DEBUG_ASSERT(is_theta_op(node->operation()));

	/* FIXME: In order to also redirect state variables,
		we need to know whether a loop terminates.*/

	jive::theta theta(node);
	for (auto & lv : theta) {
		if (lv.result()->origin() == lv.argument()
		&& dynamic_cast<const jive::value::type*>(&lv.argument()->type()))
			lv.output()->replace(lv.input()->origin());
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

		if (is_gamma_op(node->operation())) {
			gamma_invariance(strnode);
			continue;
		}

		if (is_theta_op(node->operation())) {
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
