/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/types/function/fctlambda.h>
#include <jive/view.h>
#include <jive/vsdg/control.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/phi.h>
#include <jive/vsdg/simple_node.h>
#include <jive/vsdg/theta.h>

#include <jlm/opt/cne.hpp>

static inline void
test_simple()
{
	jlm::valuetype vt;
	jlm::test_op nop({}, {&vt});
	jlm::test_op uop({&vt}, {&vt});
	jlm::test_op bop({&vt, &vt}, {&vt});

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto x = graph.import(vt, "x");
	auto y = graph.import(vt, "y");
	auto z = graph.import(vt, "z");

	auto n1 = graph.root()->add_simple_node(nop, {})->output(0);
	auto n2 = graph.root()->add_simple_node(nop, {})->output(0);

	auto u1 = graph.root()->add_simple_node(uop, {z})->output(0);

	auto b1 = graph.root()->add_simple_node(bop, {x, y})->output(0);
	auto b2 = graph.root()->add_simple_node(bop, {x, y})->output(0);
	auto b3 = graph.root()->add_simple_node(bop, {n1, z})->output(0);
	auto b4 = graph.root()->add_simple_node(bop, {n2, z})->output(0);

	graph.export_port(n1, "n1");
	graph.export_port(n2, "n2");
	graph.export_port(u1, "u1");
	graph.export_port(b1, "b1");
	graph.export_port(b2, "b2");
	graph.export_port(b3, "b3");
	graph.export_port(b4, "b4");

//	jive::view(graph.root(), stdout);
	jlm::cne(graph);
//	jive::view(graph.root(), stdout);

	assert(graph.root()->result(0)->origin() == graph.root()->result(1)->origin());
	assert(graph.root()->result(3)->origin() == graph.root()->result(4)->origin());
	assert(graph.root()->result(5)->origin() == graph.root()->result(6)->origin());
}

static inline void
test_gamma()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op nop({}, {&vt});
	jlm::test_op uop({&vt}, {&vt});

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");
	auto y = graph.import(vt, "y");
	auto z = graph.import(vt, "z");

	auto u1 = graph.root()->add_simple_node(uop, {x})->output(0);
	auto u2 = graph.root()->add_simple_node(uop, {x})->output(0);

	jive::gamma_builder gb;
	gb.begin_gamma(c);

	auto ev1 = gb.add_entryvar(u1);
	auto ev2 = gb.add_entryvar(u2);
	auto ev3 = gb.add_entryvar(y);
	auto ev4 = gb.add_entryvar(z);
	auto ev5 = gb.add_entryvar(z);

	auto n1 = gb.subregion(0)->add_simple_node(nop, {})->output(0);
	auto n2 = gb.subregion(0)->add_simple_node(nop, {})->output(0);
	auto n3 = gb.subregion(0)->add_simple_node(nop, {})->output(0);

	auto xv1 = gb.add_exitvar({ev1->argument(0), ev2->argument(1)});
	auto xv2 = gb.add_exitvar({ev2->argument(0), ev2->argument(1)});
	auto xv3 = gb.add_exitvar({ev3->argument(0), ev3->argument(1)});
	auto xv4 = gb.add_exitvar({n1, ev3->argument(1)});
	auto xv5 = gb.add_exitvar({n2, ev3->argument(1)});
	auto xv6 = gb.add_exitvar({n3, ev3->argument(1)});
	auto xv7 = gb.add_exitvar({ev5->argument(0), ev4->argument(1)});

	auto gamma = gb.end_gamma();

	graph.export_port(gamma->node()->output(0), "x1");
	graph.export_port(gamma->node()->output(1), "x2");
	graph.export_port(gamma->node()->output(2), "y");

//	jive::view(graph.root(), stdout);
	jlm::cne(graph);
//	jive::view(graph.root(), stdout);

	auto subregion0 = gamma->node()->subregion(0);
	auto subregion1 = gamma->node()->subregion(1);
	assert(gamma->node()->input(1)->origin() == gamma->node()->input(2)->origin());
	assert(subregion0->result(0)->origin() == subregion0->result(1)->origin());
	assert(subregion0->result(3)->origin() == subregion0->result(4)->origin());
	assert(subregion0->result(3)->origin() == subregion0->result(5)->origin());
	assert(subregion1->result(0)->origin() == subregion1->result(1)->origin());
	assert(graph.root()->result(0)->origin() == graph.root()->result(1)->origin());

	auto argument0 = dynamic_cast<const jive::argument*>(subregion0->result(6)->origin());
	auto argument1 = dynamic_cast<const jive::argument*>(subregion1->result(6)->origin());
	assert(argument0->input() == argument1->input());
}

static inline void
test_theta()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op uop({&vt}, {&vt});
	jlm::test_op bop({&vt, &vt}, {&vt});

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");

	jive::theta_builder tb;
	auto region = tb.begin_theta(graph.root());

	auto lv1 = tb.add_loopvar(c);
	auto lv2 = tb.add_loopvar(x);
	auto lv3 = tb.add_loopvar(x);
	auto lv4 = tb.add_loopvar(x);

	auto u1 = region->add_simple_node(uop, {lv2->argument()})->output(0);
	auto u2 = region->add_simple_node(uop, {lv3->argument()})->output(0);
	auto b1 = region->add_simple_node(bop, {lv3->argument(), lv4->argument()})->output(0);

	lv2->result()->divert_origin(u1);
	lv3->result()->divert_origin(u2);
	lv4->result()->divert_origin(b1);

	auto theta = tb.end_theta(lv1->argument());

	graph.export_port(theta->node()->output(1), "lv2");
	graph.export_port(theta->node()->output(2), "lv3");
	graph.export_port(theta->node()->output(3), "lv4");

//	jive::view(graph.root(), stdout);
	jlm::cne(graph);
//	jive::view(graph.root(), stdout);

	assert(u1->node()->input(0)->origin() == u2->node()->input(0)->origin());
	assert(region->result(2)->origin() == region->result(3)->origin());
	assert(graph.root()->result(0)->origin() == graph.root()->result(1)->origin());
}

static inline void
test_lambda()
{
	jlm::valuetype vt;
	jive::fct::type ft({&vt, &vt}, {&vt});
	jlm::test_op bop({&vt, &vt}, {&vt});

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto x = graph.import(vt, "x");

	jive::lambda_builder lb;
	auto region = lb.begin_lambda(graph.root(), ft);

	auto d1 = lb.add_dependency(x);
	auto d2 = lb.add_dependency(x);

	auto b1 = lb.subregion()->add_simple_node(bop, {d1, d2})->output(0);

	auto lambda = lb.end_lambda({b1});

	graph.export_port(lambda->node()->output(0), "f");

//	jive::view(graph.root(), stdout);
	jlm::cne(graph);
//	jive::view(graph.root(), stdout);

	assert(b1->node()->input(0)->origin() == b1->node()->input(1)->origin());
}

static inline void
test_phi()
{
	jlm::valuetype vt;
	jive::fct::type ft({&vt, &vt}, {&vt});

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto x = graph.import(vt, "x");

	jive::phi_builder pb;
	auto region = pb.begin_phi(graph.root());

	auto d1 = pb.add_dependency(x);
	auto d2 = pb.add_dependency(x);

	auto r1 = pb.add_recvar(ft);
	auto r2 = pb.add_recvar(ft);

	jive::lambda_builder lb;
	lb.begin_lambda(region, ft);
	d1 = lb.add_dependency(d1);
	auto f1 = lb.end_lambda({d1});

	lb.begin_lambda(region, ft);
	d2 = lb.add_dependency(d2);
	auto f2 = lb.end_lambda({d2});

	r1->set_value(f1->node()->output(0));
	r2->set_value(f2->node()->output(0));

	auto phi = pb.end_phi();

	graph.export_port(phi->output(0), "f1");
	graph.export_port(phi->output(1), "f2");

//	jive::view(graph.root(), stdout);
	jlm::cne(graph);
//	jive::view(graph.root(), stdout);

	assert(f1->node()->input(0)->origin() == f2->node()->input(0)->origin());
}

static int
verify()
{
	test_simple();
	test_gamma();
	test_theta();
	test_lambda();
	test_phi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-cne", verify);
