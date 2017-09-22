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

#include <jlm/opt/push.hpp>

static inline void
test_gamma()
{
	jlm::statetype st;
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op nop({}, {&vt});
	jlm::test_op sop({&vt, &st}, {&st});
	jlm::test_op binop({&vt, &vt}, {&vt});

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");
	auto s = graph.import(st, "s");

	jive::gamma_builder gb;
	gb.begin_gamma(c);
	auto evx = gb.add_entryvar(x);
	auto evs = gb.add_entryvar(s);

	auto null = gb.subregion(0)->add_simple_node(nop, {})->output(0);
	auto bin = gb.subregion(0)->add_simple_node(binop, {null, evx->argument(0)})->output(0);
	auto state = gb.subregion(0)->add_simple_node(sop, {bin, evs->argument(0)})->output(0);

	auto xvs = gb.add_exitvar({state, evs->argument(1)});
	auto gamma = gb.end_gamma();

	graph.export_port(gamma->node()->output(0), "x");

	jive::view(graph.root(), stdout);
	jlm::push(graph);
	jive::view(graph.root(), stdout);

	assert(graph.root()->nodes.size() == 3);
}

static inline void
test_theta()
{
	jlm::statetype st;
	jlm::valuetype vt;
	jive::ctl::type ct(2);

	jlm::test_op nop({}, {&vt});
	jlm::test_op bop({&vt, &vt}, {&vt});
	jlm::test_op sop({&vt, &st}, {&st});

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");
	auto s = graph.import(st, "s");

	jive::theta_builder tb;
	tb.begin_theta(graph.root());

	auto lv1 = tb.add_loopvar(c);
	auto lv2 = tb.add_loopvar(x);
	auto lv3 = tb.add_loopvar(x);
	auto lv4 = tb.add_loopvar(s);

	auto o1 = tb.subregion()->add_simple_node(nop, {})->output(0);
	auto o2 = tb.subregion()->add_simple_node(bop, {o1, lv3->argument()})->output(0);
	auto o3 = tb.subregion()->add_simple_node(bop, {lv2->argument(), o2})->output(0);
	auto o4 = tb.subregion()->add_simple_node(sop, {lv3->argument(), lv4->argument()})->output(0);

	lv2->result()->divert_origin(o3);
	lv4->result()->divert_origin(o4);

	auto theta = tb.end_theta(lv1->argument());

	graph.export_port(theta->node()->output(0), "c");

	jive::view(graph.root(), stdout);
	jlm::push(graph);
	jive::view(graph.root(), stdout);

	assert(graph.root()->nodes.size() == 3);
}

static int
verify()
{
	test_gamma();
	test_theta();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-push", verify);
