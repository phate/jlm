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
#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/phi.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/theta.h>

#include <jlm/ir/operators/lambda.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/cne.hpp>
#include <jlm/util/stats.hpp>

static const jlm::stats_descriptor sd;

static inline void
test_simple()
{
	using namespace jlm;

	jlm::valuetype vt;

	rvsdg_module rm(filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});
	auto z = graph.add_import({vt, "z"});

	auto n1 = jlm::create_testop(graph.root(), {}, {&vt})[0];
	auto n2 = jlm::create_testop(graph.root(), {}, {&vt})[0];

	auto u1 = jlm::create_testop(graph.root(), {z}, {&vt})[0];

	auto b1 = jlm::create_testop(graph.root(), {x, y}, {&vt})[0];
	auto b2 = jlm::create_testop(graph.root(), {x, y}, {&vt})[0];
	auto b3 = jlm::create_testop(graph.root(), {n1, z}, {&vt})[0];
	auto b4 = jlm::create_testop(graph.root(), {n2, z}, {&vt})[0];

	graph.add_export(n1, {n1->type(), "n1"});
	graph.add_export(n2, {n2->type(), "n2"});
	graph.add_export(u1, {n2->type(), "u1"});
	graph.add_export(b1, {n2->type(), "b1"});
	graph.add_export(b2, {n2->type(), "b2"});
	graph.add_export(b3, {n2->type(), "b3"});
	graph.add_export(b4, {n2->type(), "b4"});

//	jive::view(graph.root(), stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph.root(), stdout);

	assert(graph.root()->result(0)->origin() == graph.root()->result(1)->origin());
	assert(graph.root()->result(3)->origin() == graph.root()->result(4)->origin());
	assert(graph.root()->result(5)->origin() == graph.root()->result(6)->origin());
}

static inline void
test_gamma()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	rvsdg_module rm(filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});
	auto z = graph.add_import({vt, "z"});

	auto u1 = jlm::create_testop(graph.root(), {x}, {&vt})[0];
	auto u2 = jlm::create_testop(graph.root(), {x}, {&vt})[0];

	auto gamma = jive::gamma_node::create(c, 2);

	auto ev1 = gamma->add_entryvar(u1);
	auto ev2 = gamma->add_entryvar(u2);
	auto ev3 = gamma->add_entryvar(y);
	auto ev4 = gamma->add_entryvar(z);
	auto ev5 = gamma->add_entryvar(z);

	auto n1 = jlm::create_testop(gamma->subregion(0), {}, {&vt})[0];
	auto n2 = jlm::create_testop(gamma->subregion(0), {}, {&vt})[0];
	auto n3 = jlm::create_testop(gamma->subregion(0), {}, {&vt})[0];

	gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});
	gamma->add_exitvar({ev2->argument(0), ev2->argument(1)});
	gamma->add_exitvar({ev3->argument(0), ev3->argument(1)});
	gamma->add_exitvar({n1, ev3->argument(1)});
	gamma->add_exitvar({n2, ev3->argument(1)});
	gamma->add_exitvar({n3, ev3->argument(1)});
	gamma->add_exitvar({ev5->argument(0), ev4->argument(1)});

	graph.add_export(gamma->output(0), {gamma->output(0)->type(), "x1"});
	graph.add_export(gamma->output(1), {gamma->output(1)->type(), "x2"});
	graph.add_export(gamma->output(2), {gamma->output(2)->type(), "y"});

//	jive::view(graph.root(), stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph.root(), stdout);

	auto subregion0 = gamma->subregion(0);
	auto subregion1 = gamma->subregion(1);
	assert(gamma->input(1)->origin() == gamma->input(2)->origin());
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
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	rvsdg_module rm(filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});

	auto theta = jive::theta_node::create(graph.root());
	auto region = theta->subregion();

	auto lv1 = theta->add_loopvar(c);
	auto lv2 = theta->add_loopvar(x);
	auto lv3 = theta->add_loopvar(x);
	auto lv4 = theta->add_loopvar(x);

	auto u1 = jlm::create_testop(region, {lv2->argument()}, {&vt})[0];
	auto u2 = jlm::create_testop(region, {lv3->argument()}, {&vt})[0];
	auto b1 = jlm::create_testop(region, {lv3->argument(), lv4->argument()}, {&vt})[0];

	lv2->result()->divert_to(u1);
	lv3->result()->divert_to(u2);
	lv4->result()->divert_to(b1);

	theta->set_predicate(lv1->argument());

	graph.add_export(theta->output(1), {theta->output(1)->type(), "lv2"});
	graph.add_export(theta->output(2), {theta->output(2)->type(), "lv3"});
	graph.add_export(theta->output(3), {theta->output(3)->type(), "lv4"});

//	jive::view(graph.root(), stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph.root(), stdout);

	assert(u1->node()->input(0)->origin() == u2->node()->input(0)->origin());
	assert(b1->node()->input(0)->origin() == u1->node()->input(0)->origin());
	assert(b1->node()->input(1)->origin() == region->argument(3));
	assert(region->result(2)->origin() == region->result(3)->origin());
	assert(graph.root()->result(0)->origin() == graph.root()->result(1)->origin());
}

static inline void
test_theta2()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	rvsdg_module rm(jlm::filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});

	auto theta = jive::theta_node::create(graph.root());
	auto region = theta->subregion();

	auto lv1 = theta->add_loopvar(c);
	auto lv2 = theta->add_loopvar(x);
	auto lv3 = theta->add_loopvar(x);

	auto u1 = jlm::create_testop(region, {lv2->argument()}, {&vt})[0];
	auto u2 = jlm::create_testop(region, {lv3->argument()}, {&vt})[0];
	auto b1 = jlm::create_testop(region, {u2, u2}, {&vt})[0];

	lv2->result()->divert_to(u1);
	lv3->result()->divert_to(b1);

	theta->set_predicate(lv1->argument());

	graph.add_export(theta->output(1), {theta->output(1)->type(), "lv2"});
	graph.add_export(theta->output(2), {theta->output(2)->type(), "lv3"});

//	jive::view(graph, stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph, stdout);

	assert(lv2->result()->origin() == u1);
	assert(lv2->argument()->nusers() != 0 && lv3->argument()->nusers() != 0);
}

static inline void
test_theta3()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	rvsdg_module rm(jlm::filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});

	auto theta1 = jive::theta_node::create(graph.root());
	auto r1 = theta1->subregion();

	auto lv1 = theta1->add_loopvar(c);
	auto lv2 = theta1->add_loopvar(x);
	auto lv3 = theta1->add_loopvar(x);
	auto lv4 = theta1->add_loopvar(x);

	auto theta2 = jive::theta_node::create(r1);
	auto r2 = theta2->subregion();
	auto p = theta2->add_loopvar(lv1->argument());
	theta2->add_loopvar(lv2->argument());
	theta2->add_loopvar(lv3->argument());
	theta2->add_loopvar(lv4->argument());
	theta2->set_predicate(p->argument());

	auto u1 = jlm::create_testop(r1, {theta2->output(1)}, {&vt})[0]->node();
	auto b1 = jlm::create_testop(r1, {theta2->output(2), theta2->output(2)}, {&vt})[0]->node();
	auto u2 = jlm::create_testop(r1, {theta2->output(3)}, {&vt})[0]->node();

	lv2->result()->divert_to(u1->output(0));
	lv3->result()->divert_to(b1->output(0));
	lv4->result()->divert_to(u1->output(0));

	theta1->set_predicate(lv1->argument());

	graph.add_export(theta1->output(1), {theta1->output(1)->type(), "lv2"});
	graph.add_export(theta1->output(2), {theta1->output(2)->type(), "lv3"});
	graph.add_export(theta1->output(3), {theta1->output(3)->type(), "lv4"});

//	jive::view(graph, stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph, stdout);

	assert(r1->result(2)->origin() == r1->result(4)->origin());
	assert(u1->input(0)->origin() == u2->input(0)->origin());
	assert(r2->result(2)->origin() == r2->result(4)->origin());
	assert(theta2->input(1)->origin() == theta2->input(3)->origin());
	assert(r1->result(3)->origin() != r1->result(4)->origin());
	assert(r2->result(3)->origin() != r2->result(4)->origin());
}

static inline void
test_theta4()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	rvsdg_module rm(jlm::filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});

	auto theta = jive::theta_node::create(graph.root());
	auto region = theta->subregion();

	auto lv1 = theta->add_loopvar(c);
	auto lv2 = theta->add_loopvar(x);
	auto lv3 = theta->add_loopvar(x);
	auto lv4 = theta->add_loopvar(y);
	auto lv5 = theta->add_loopvar(y);
	auto lv6 = theta->add_loopvar(x);
	auto lv7 = theta->add_loopvar(x);

	auto u1 = jlm::create_testop(region, {lv2->argument()}, {&vt})[0]->node();
	auto b1 = jlm::create_testop(region, {lv3->argument(), lv3->argument()}, {&vt})[0]->node();

	lv2->result()->divert_to(lv4->argument());
	lv3->result()->divert_to(lv5->argument());
	lv4->result()->divert_to(u1->output(0));
	lv5->result()->divert_to(b1->output(0));

	theta->set_predicate(lv1->argument());

	auto ex1 = graph.add_export(theta->output(1), {theta->output(1)->type(), "lv2"});
	auto ex2 = graph.add_export(theta->output(2), {theta->output(2)->type(), "lv3"});
	graph.add_export(theta->output(3), {theta->output(3)->type(), "lv4"});
	graph.add_export(theta->output(4), {theta->output(4)->type(), "lv5"});

//	jive::view(graph, stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph, stdout);

	assert(ex1->origin() != ex2->origin());
	assert(lv2->argument()->nusers() != 0 && lv3->argument()->nusers() != 0);
	assert(lv6->result()->origin() == lv7->result()->origin());
}

static inline void
test_theta5()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	rvsdg_module rm(jlm::filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});

	auto theta = jive::theta_node::create(graph.root());
	auto region = theta->subregion();

	auto lv0 = theta->add_loopvar(c);
	auto lv1 = theta->add_loopvar(x);
	auto lv2 = theta->add_loopvar(x);
	auto lv3 = theta->add_loopvar(y);
	auto lv4 = theta->add_loopvar(y);

	lv1->result()->divert_to(lv3->argument());
	lv2->result()->divert_to(lv4->argument());

	theta->set_predicate(lv0->argument());

	auto ex1 = graph.add_export(theta->output(1), {theta->output(1)->type(), "lv1"});
	auto ex2 = graph.add_export(theta->output(2), {theta->output(2)->type(), "lv2"});
	auto ex3 = graph.add_export(theta->output(3), {theta->output(3)->type(), "lv3"});
	auto ex4 = graph.add_export(theta->output(4), {theta->output(4)->type(), "lv4"});

//	jive::view(graph, stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph, stdout);

	assert(ex1->origin() == ex2->origin());
	assert(ex3->origin() == ex4->origin());
	assert(region->result(4)->origin() == region->result(5)->origin());
	assert(region->result(2)->origin() == region->result(3)->origin());
}

static inline void
test_lambda()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::fcttype ft({&vt, &vt}, {&vt});

	rvsdg_module rm(jlm::filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto x = graph.add_import({vt, "x"});

	jlm::lambda_builder lb;
	auto region = lb.begin_lambda(graph.root(), {ft, "f", linkage::external_linkage});

	auto d1 = lb.add_dependency(x);
	auto d2 = lb.add_dependency(x);

	auto b1 = jlm::create_testop(lb.subregion(), {d1, d2}, {&vt})[0];

	auto lambda = lb.end_lambda({b1});

	graph.add_export(lambda->output(0), {lambda->output(0)->type(), "f"});

//	jive::view(graph.root(), stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph.root(), stdout);

	assert(b1->node()->input(0)->origin() == b1->node()->input(1)->origin());
}

static inline void
test_phi()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::fcttype ft({&vt, &vt}, {&vt});

	rvsdg_module rm(jlm::filepath(""), "", "");
	auto & graph = *rm.graph();
	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto x = graph.add_import({vt, "x"});

	jive::phi_builder pb;
	auto region = pb.begin_phi(graph.root());

	auto d1 = pb.add_dependency(x);
	auto d2 = pb.add_dependency(x);

	auto r1 = pb.add_recvar(ft);
	auto r2 = pb.add_recvar(ft);

	jlm::lambda_builder lb;
	lb.begin_lambda(region, {ft, "f", linkage::external_linkage});
	d1 = lb.add_dependency(d1);
	auto f1 = lb.end_lambda({d1});

	lb.begin_lambda(region, {ft, "g", linkage::external_linkage});
	d2 = lb.add_dependency(d2);
	auto f2 = lb.end_lambda({d2});

	r1->set_value(f1->output(0));
	r2->set_value(f2->output(0));

	auto phi = pb.end_phi();

	graph.add_export(phi->output(0), {phi->output(0)->type(), "f1"});
	graph.add_export(phi->output(1), {phi->output(1)->type(), "f2"});

//	jive::view(graph.root(), stdout);
	jlm::cne cne;
	cne.run(rm, sd);
//	jive::view(graph.root(), stdout);

	assert(f1->input(0)->origin() == f2->input(0)->origin());
}

static int
verify()
{
	test_simple();
	test_gamma();
	test_theta();
	test_theta2();
	test_theta3();
	test_theta4();
	test_theta5();
	test_lambda();
	test_phi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-cne", verify)
