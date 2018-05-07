/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/view.h>
#include <jive/rvsdg/control.h>
#include <jive/rvsdg/gamma.h>

#include <jlm/ir/operators.hpp>
#include <jlm/opt/inlining.hpp>

/* FIXME: replace with contains function from jive */
static bool
contains_call_node(const jive::region * region)
{
	for (const auto & node : region->nodes) {
		if (jive::is<jlm::call_op>(&node))
			return true;

		if (auto structnode = dynamic_cast<const jive::structural_node*>(&node)) {
			for (size_t n = 0; n < structnode->nsubregions(); n++) {
				if (contains_call_node(structnode->subregion(n)))
					return true;
			}
		}
	}

	return false;
}

static int
verify()
{
	jlm::valuetype vt;
	jive::ctltype ct(2);
	jive::fcttype ft1({&vt}, {&vt});
	jive::fcttype ft2({&ct, &vt}, {&vt});

	jive::graph graph;
	auto i = graph.add_import(vt, "i");

	/* f1 */
	jlm::lambda_builder lb;
	auto arguments = lb.begin_lambda(graph.root(), {ft1, "f1"});
	lb.add_dependency(i);
	auto t = jlm::create_testop(lb.subregion(), {arguments[0]}, {&vt})[0]->node();
	auto f1 = lb.end_lambda({t->output(0)});

	/* f2 */
	arguments = lb.begin_lambda(graph.root(), {ft2, "f2"});
	auto d = lb.add_dependency(f1->output(0));

	auto gamma = jive::gamma_node::create(arguments[0], 2);
	auto ev1 = gamma->add_entryvar(arguments[1]);
	auto ev2 = gamma->add_entryvar(d);
	auto apply = jlm::create_call(ev2->argument(0), {ev1->argument(0)})[0];
	auto xv1 = gamma->add_exitvar({apply, ev1->argument(1)});
	auto f2 = lb.end_lambda({xv1});

	graph.add_export(f2->output(0), "f2");

	jive::view(graph.root(), stdout);
	jlm::inlining(graph);
	jive::view(graph.root(), stdout);

	assert(!contains_call_node(graph.root()));
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-inlining", verify);
