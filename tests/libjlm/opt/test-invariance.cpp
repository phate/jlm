/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/view.h>
#include <jive/rvsdg/controltype.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/theta.h>

#include <jlm/opt/invariance.hpp>

static inline void
test_gamma()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");
	auto y = graph.add_import(vt, "y");

	auto gamma1 = jive::gamma_node::create(c, 2);
	auto ev1 = gamma1->add_entryvar(c);
	auto ev2 = gamma1->add_entryvar(x);
	auto ev3 = gamma1->add_entryvar(y);

	auto gamma2 = jive::gamma_node::create(ev1->argument(0), 2);
	auto ev4 = gamma2->add_entryvar(ev2->argument(0));
	auto ev5 = gamma2->add_entryvar(ev3->argument(0));
	gamma2->add_exitvar({ev4->argument(0), ev4->argument(1)});
	gamma2->add_exitvar({ev5->argument(0), ev5->argument(1)});

	gamma1->add_exitvar({gamma2->output(0), ev2->argument(1)});
	gamma1->add_exitvar({gamma2->output(1), ev3->argument(1)});

	graph.add_export(gamma1->output(0), "x");
	graph.add_export(gamma1->output(1), "y");

	jive::view(graph.root(), stdout);
	jlm::invariance(graph);
	jive::view(graph.root(), stdout);

	assert(graph.root()->result(0)->origin() == graph.root()->argument(1));
	assert(graph.root()->result(1)->origin() == graph.root()->argument(2));
}

static inline void
test_theta()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");

	auto theta1 = jive::theta_node::create(graph.root());
	auto lv1 = theta1->add_loopvar(c);
	auto lv2 = theta1->add_loopvar(x);

	auto theta2 = jive::theta_node::create(theta1->subregion());
	auto lv3 = theta2->add_loopvar(lv1->argument());
	theta2->add_loopvar(lv2->argument());
	theta2->set_predicate(lv3->argument());

	theta1->set_predicate(lv1->argument());

	graph.add_export(lv1, "c");
	graph.add_export(lv2, "x");

	jive::view(graph.root(), stdout);
	jlm::invariance(graph);
	jive::view(graph.root(), stdout);

	assert(graph.root()->result(0)->origin() == lv1);
	assert(graph.root()->result(1)->origin() == graph.root()->argument(1));
}

static int
verify()
{
	test_gamma();
	test_theta();
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-invariance", verify);
