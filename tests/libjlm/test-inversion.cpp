/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/view.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/simple_node.h>
#include <jive/vsdg/theta.h>

#include <jlm/opt/inversion.hpp>

static inline void
test1()
{
	jlm::valuetype vt;
	jive::bits::type bt(1);
	jlm::test_op bop1({&vt, &vt}, {&bt});
	jlm::test_op bop2({&vt, &vt}, {&vt});

	jive::graph graph;

	auto x = graph.import(vt, "x");
	auto y = graph.import(vt, "y");
	auto z = graph.import(vt, "z");

	jive::theta_builder tb;
	tb.begin_theta(graph.root());

	auto lvx = tb.add_loopvar(x);
	auto lvy = tb.add_loopvar(y);
	auto lvz = tb.add_loopvar(z);

	auto a = tb.subregion()->add_simple_node(bop1, {lvx->argument(), lvy->argument()});
	auto predicate = jive::ctl::match(1, {{1, 0}}, 1, 2, a->output(0));

	jive::gamma_builder gb;
	gb.begin_gamma(predicate);

	auto evx = gb.add_entryvar(lvx->argument());
	auto evy = gb.add_entryvar(lvy->argument());

	auto b = gb.subregion(0)->add_simple_node(bop2, {evx->argument(0), evy->argument(0)});
	auto c = gb.subregion(1)->add_simple_node(bop2, {evx->argument(1), evy->argument(1)});

	auto xvy = gb.add_exitvar({b->output(0), c->output(0)});
	gb.end_gamma();

	lvy->result()->divert_origin(xvy->output());

	auto theta = tb.end_theta(predicate);

	auto ex1 = graph.export_port(theta->node()->output(0), "x");
	auto ex2 = graph.export_port(theta->node()->output(1), "y");
	auto ex3 = graph.export_port(theta->node()->output(2), "z");

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

	auto x = graph.import(vt, "x");

	jive::theta_builder tb;
	tb.begin_theta(graph.root());

	auto lv1 = tb.add_loopvar(x);

	auto n1 = tb.subregion()->add_simple_node(cop, {lv1->argument()});
	auto n2 = tb.subregion()->add_simple_node(uop, {lv1->argument()});
	auto predicate = jive::ctl::match(1, {{1, 0}}, 1, 2, n1->output(0));

	jive::gamma_builder gb;
	gb.begin_gamma(predicate);

	auto ev1 = gb.add_entryvar(n1->output(0));
	auto ev2 = gb.add_entryvar(lv1->argument());
	auto ev3 = gb.add_entryvar(n2->output(0));

	auto xv1 = gb.add_exitvar({ev1->argument(0), ev1->argument(1)});
	auto xv2 = gb.add_exitvar({ev2->argument(0), ev2->argument(1)});
	auto xv3 = gb.add_exitvar({ev3->argument(0), ev3->argument(1)});

	auto gamma = gb.end_gamma();

	lv1->result()->divert_origin(gamma->node()->output(1));

	auto theta = tb.end_theta(predicate);

	auto ex = graph.export_port(theta->node()->output(0), "x");

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

JLM_UNIT_TEST_REGISTER("libjlm/test-inversion", verify);
