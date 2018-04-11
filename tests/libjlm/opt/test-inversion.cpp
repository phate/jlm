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

#include <jlm/opt/inversion.hpp>

static inline void
test1()
{
	jlm::valuetype vt;
	jive::bits::type bt(1);
	jlm::test_op bop1({&vt, &vt}, {&bt});
	jlm::test_op bop2({&vt, &vt}, {&vt});

	jive::graph graph;

	auto x = graph.add_import(vt, "x");
	auto y = graph.add_import(vt, "y");
	auto z = graph.add_import(vt, "z");

	auto theta = jive::theta_node::create(graph.root());

	auto lvx = theta->add_loopvar(x);
	auto lvy = theta->add_loopvar(y);
	theta->add_loopvar(z);

	auto a = theta->subregion()->add_simple_node(bop1, {lvx->argument(), lvy->argument()});
	auto predicate = jive::ctl::match(1, {{1, 0}}, 1, 2, a->output(0));

	auto gamma = jive::gamma_node::create(predicate, 2);

	auto evx = gamma->add_entryvar(lvx->argument());
	auto evy = gamma->add_entryvar(lvy->argument());

	auto b = gamma->subregion(0)->add_simple_node(bop2, {evx->argument(0), evy->argument(0)});
	auto c = gamma->subregion(1)->add_simple_node(bop2, {evx->argument(1), evy->argument(1)});

	auto xvy = gamma->add_exitvar({b->output(0), c->output(0)});

	lvy->result()->divert_origin(xvy->output());

	theta->set_predicate(predicate);

	auto ex1 = graph.add_export(theta->output(0), "x");
	auto ex2 = graph.add_export(theta->output(1), "y");
	auto ex3 = graph.add_export(theta->output(2), "z");

//	jive::view(graph.root(), stdout);
	jlm::invert(graph);
//	jive::view(graph.root(), stdout);

	assert(is_gamma_node(ex1->origin()->node()));
	assert(is_gamma_node(ex2->origin()->node()));
	assert(is_gamma_node(ex3->origin()->node()));
}

static inline void
test2()
{
	jlm::valuetype vt;
	jive::bits::type bt1(1);
	jlm::test_op cop({&vt}, {&bt1});
	jlm::test_op uop({&vt}, {&vt});

	jive::graph graph;

	auto x = graph.add_import(vt, "x");

	auto theta = jive::theta_node::create(graph.root());

	auto lv1 = theta->add_loopvar(x);

	auto n1 = theta->subregion()->add_simple_node(cop, {lv1->argument()});
	auto n2 = theta->subregion()->add_simple_node(uop, {lv1->argument()});
	auto predicate = jive::ctl::match(1, {{1, 0}}, 1, 2, n1->output(0));

	auto gamma = jive::gamma_node::create(predicate, 2);

	auto ev1 = gamma->add_entryvar(n1->output(0));
	auto ev2 = gamma->add_entryvar(lv1->argument());
	auto ev3 = gamma->add_entryvar(n2->output(0));

	auto xv1 = gamma->add_exitvar({ev1->argument(0), ev1->argument(1)});
	auto xv2 = gamma->add_exitvar({ev2->argument(0), ev2->argument(1)});
	auto xv3 = gamma->add_exitvar({ev3->argument(0), ev3->argument(1)});

	lv1->result()->divert_origin(gamma->output(1));

	theta->set_predicate(predicate);

	auto ex = graph.add_export(theta->output(0), "x");

//	jive::view(graph.root(), stdout);
	jlm::invert(graph);
//	jive::view(graph.root(), stdout);

	assert(is_gamma_node(ex->origin()->node()));
}

static int
verify()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-inversion", verify);
