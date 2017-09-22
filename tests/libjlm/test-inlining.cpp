/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/types/function/fctapply.h>
#include <jive/types/function/fctlambda.h>
#include <jive/view.h>
#include <jive/vsdg/control.h>

#include <jlm/opt/inlining.hpp>

static int
verify()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jive::fct::type ft1({&vt}, {&vt});
	jive::fct::type ft2({&ct, &vt}, {&vt});
	jlm::test_op op({&vt}, {&vt});

	jive::graph graph;
	auto i = graph.import(vt, "i");

	/* f1 */
	jive::lambda_builder lb;
	auto arguments = lb.begin_lambda(graph.root(), ft1);
	lb.add_dependency(i);
	auto t = lb.subregion()->add_simple_node(op, arguments);
	auto f1 = lb.end_lambda({t->output(0)});

	/* f2 */
	arguments = lb.begin_lambda(graph.root(), ft2);
	auto d = lb.add_dependency(f1->node()->output(0));

	jive::gamma_builder gb;
	gb.begin_gamma(arguments[0]);
	auto ev1 = gb.add_entryvar(arguments[1]);
	auto ev2 = gb.add_entryvar(d);
	auto apply = jive::fct::create_apply(ev2->argument(0), {ev1->argument(0)})[0];
	auto xv1 = gb.add_exitvar({apply, ev1->argument(1)});
	gb.end_gamma();
	auto f2 = lb.end_lambda({xv1->output()});

	graph.export_port(f2->node()->output(0), "f2");

	jive::view(graph.root(), stdout);

	jlm::inlining(graph);

	jive::view(graph.root(), stdout);

	assert(apply->nusers() == 0);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-inlining", verify);
