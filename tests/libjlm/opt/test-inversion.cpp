/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/view.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/theta.h>

#include <jlm/jlm/opt/inversion.hpp>

static inline void
test1()
{
	jlm::valuetype vt;
	jive::bittype bt(1);

	jive::graph graph;

	auto x = graph.add_import(vt, "x");
	auto y = graph.add_import(vt, "y");
	auto z = graph.add_import(vt, "z");

	auto theta = jive::theta_node::create(graph.root());

	auto lvx = theta->add_loopvar(x);
	auto lvy = theta->add_loopvar(y);
	theta->add_loopvar(z);

	auto a = jlm::create_testop(theta->subregion(), {lvx->argument(), lvy->argument()}, {&bt})[0];
	auto predicate = jive::match(1, {{1, 0}}, 1, 2, a);

	auto gamma = jive::gamma_node::create(predicate, 2);

	auto evx = gamma->add_entryvar(lvx->argument());
	auto evy = gamma->add_entryvar(lvy->argument());

	auto b = jlm::create_testop(gamma->subregion(0), {evx->argument(0), evy->argument(0)}, {&vt})[0];
	auto c = jlm::create_testop(gamma->subregion(1), {evx->argument(1), evy->argument(1)}, {&vt})[0];

	auto xvy = gamma->add_exitvar({b, c});

	lvy->result()->divert_to(xvy);

	theta->set_predicate(predicate);

	auto ex1 = graph.add_export(theta->output(0), "x");
	auto ex2 = graph.add_export(theta->output(1), "y");
	auto ex3 = graph.add_export(theta->output(2), "z");

//	jive::view(graph.root(), stdout);
	jlm::invert(graph);
//	jive::view(graph.root(), stdout);

	assert(jive::is<jive::gamma_op>(ex1->origin()->node()));
	assert(jive::is<jive::gamma_op>(ex2->origin()->node()));
	assert(jive::is<jive::gamma_op>(ex3->origin()->node()));
}

static inline void
test2()
{
	jlm::valuetype vt;

	jive::graph graph;

	auto x = graph.add_import(vt, "x");

	auto theta = jive::theta_node::create(graph.root());

	auto lv1 = theta->add_loopvar(x);

	auto n1 = jlm::create_testop(theta->subregion(), {lv1->argument()}, {&jive::bit1})[0];
	auto n2 = jlm::create_testop(theta->subregion(), {lv1->argument()}, {&vt})[0];
	auto predicate = jive::match(1, {{1, 0}}, 1, 2, n1);

	auto gamma = jive::gamma_node::create(predicate, 2);

	auto ev1 = gamma->add_entryvar(n1);
	auto ev2 = gamma->add_entryvar(lv1->argument());
	auto ev3 = gamma->add_entryvar(n2);

	gamma->add_exitvar({ev1->argument(0), ev1->argument(1)});
	gamma->add_exitvar({ev2->argument(0), ev2->argument(1)});
	gamma->add_exitvar({ev3->argument(0), ev3->argument(1)});

	lv1->result()->divert_to(gamma->output(1));

	theta->set_predicate(predicate);

	auto ex = graph.add_export(theta->output(0), "x");

//	jive::view(graph.root(), stdout);
	jlm::invert(graph);
//	jive::view(graph.root(), stdout);

	assert(jive::is<jive::gamma_op>(ex->origin()->node()));
}

static int
verify()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-inversion", verify);
