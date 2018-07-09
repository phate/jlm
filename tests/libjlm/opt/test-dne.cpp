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

#include <jlm/jlm/ir/operators/lambda.hpp>
#include <jlm/jlm/opt/dne.hpp>

static inline void
test_root()
{
	jive::graph graph;
	graph.add_import(jlm::valuetype(), "x");
	auto y = graph.add_import(jlm::valuetype(), "y");
	graph.add_export(y, "z");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(graph.root()->narguments() == 1);
}

static inline void
test_gamma()
{
	jlm::valuetype vt;
	jive::ctltype ct(2);

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");
	auto y = graph.add_import(vt, "y");

	auto gamma = jive::gamma_node::create(c, 2);
	auto ev1 = gamma->add_entryvar(x);
	auto ev2 = gamma->add_entryvar(y);
	auto ev3 = gamma->add_entryvar(x);

	auto t = jlm::create_testop(gamma->subregion(1), {ev2->argument(1)}, {&vt})[0];

	gamma->add_exitvar({ev1->argument(0), ev1->argument(1)});
	gamma->add_exitvar({ev2->argument(0), t});
	gamma->add_exitvar({ev3->argument(0), ev1->argument(1)});

	graph.add_export(gamma->output(0), "z");
	graph.add_export(gamma->output(2), "w");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(gamma->noutputs() == 2);
	assert(gamma->subregion(1)->nodes.size() == 0);
	assert(gamma->subregion(1)->narguments() == 2);
	assert(gamma->ninputs() == 3);
	assert(graph.root()->narguments() == 2);
}

static inline void
test_gamma2()
{
	jlm::valuetype vt;
	jive::ctltype ct(2);

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");

	auto gamma = jive::gamma_node::create(c, 2);
	gamma->add_entryvar(x);

	auto n1 = jlm::create_testop(gamma->subregion(0), {}, {&vt})[0];
	auto n2 = jlm::create_testop(gamma->subregion(1), {}, {&vt})[0];

	gamma->add_exitvar({n1, n2});

	graph.add_export(gamma->output(0), "x");

//	jive::view(graph, stdout);
	jlm::dne(graph);
//	jive::view(graph, stdout);

	assert(graph.root()->narguments() == 1);
}

static inline void
test_theta()
{
	jlm::valuetype vt;
	jive::ctltype ct(2);

	jive::graph graph;
	auto x = graph.add_import(vt, "x");
	auto y = graph.add_import(vt, "y");
	auto z = graph.add_import(vt, "z");

	auto theta = jive::theta_node::create(graph.root());

	auto lv1 = theta->add_loopvar(x);
	auto lv2 = theta->add_loopvar(y);
	auto lv3 = theta->add_loopvar(z);
	auto lv4 = theta->add_loopvar(y);

	lv1->result()->divert_to(lv2->argument());
	lv2->result()->divert_to(lv1->argument());

	auto t = jlm::create_testop(theta->subregion(), {lv3->argument()}, {&vt})[0];
	lv3->result()->divert_to(t);
	lv4->result()->divert_to(lv2->argument());

	auto c = jlm::create_testop(theta->subregion(), {}, {&ct})[0];
	theta->set_predicate(c);

	graph.add_export(theta->output(0), "a");
	graph.add_export(theta->output(3), "b");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(theta->noutputs() == 3);
	assert(theta->subregion()->nodes.size() == 1);
	assert(graph.root()->narguments() == 2);
}

static inline void
test_nested_theta()
{
	jlm::valuetype vt;
	jive::ctltype ct(2);

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");
	auto y = graph.add_import(vt, "y");

	auto otheta = jive::theta_node::create(graph.root());

	auto lvo1 = otheta->add_loopvar(c);
	auto lvo2 = otheta->add_loopvar(x);
	auto lvo3 = otheta->add_loopvar(y);

	auto itheta = jive::theta_node::create(otheta->subregion());

	auto lvi1 = itheta->add_loopvar(lvo1->argument());
	auto lvi2 = itheta->add_loopvar(lvo2->argument());
	auto lvi3 = itheta->add_loopvar(lvo3->argument());

	lvi2->result()->divert_to(lvi3->argument());

	itheta->set_predicate(lvi1->argument());

	lvo2->result()->divert_to(itheta->output(1));
	lvo3->result()->divert_to(itheta->output(1));

	otheta->set_predicate(lvo1->argument());

	graph.add_export(otheta->output(2), "y");

//	jive::view(graph, stdout);
	jlm::dne(graph);
//	jive::view(graph, stdout);

	assert(otheta->noutputs() == 3);
}

static inline void
test_evolving_theta()
{
	jlm::valuetype vt;
	jive::ctltype ct(2);

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x1 = graph.add_import(vt, "x1");
	auto x2 = graph.add_import(vt, "x2");
	auto x3 = graph.add_import(vt, "x3");
	auto x4 = graph.add_import(vt, "x4");

	auto theta = jive::theta_node::create(graph.root());

	auto lv0 = theta->add_loopvar(c);
	auto lv1 = theta->add_loopvar(x1);
	auto lv2 = theta->add_loopvar(x2);
	auto lv3 = theta->add_loopvar(x3);
	auto lv4 = theta->add_loopvar(x4);

	lv1->result()->divert_to(lv2->argument());
	lv2->result()->divert_to(lv3->argument());
	lv3->result()->divert_to(lv4->argument());

	theta->set_predicate(lv0->argument());

	graph.add_export(lv1, "x1");

//	jive::view(graph, stdout);
	jlm::dne(graph);
//	jive::view(graph, stdout);

	assert(theta->noutputs() == 5);
}

static inline void
test_lambda()
{
	using namespace jlm;

	jlm::valuetype vt;

	jive::graph graph;
	auto x = graph.add_import(vt, "x");

	jlm::lambda_builder lb;
	auto arguments = lb.begin_lambda(graph.root(), {{{&vt}, {&vt}}, "f", linkage::external_linkage});

	auto d = lb.add_dependency(x);
	jlm::create_testop(lb.subregion(), {arguments[0], d}, {&vt});

	auto lambda = lb.end_lambda({arguments[0]});

	graph.add_export(lambda->output(0), "f");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(lambda->subregion()->nodes.size() == 0);
	assert(graph.root()->narguments() == 0);
}

static inline void
test_phi()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::fcttype ft({&vt}, {&vt});

	jive::graph graph;
	auto x = graph.add_import(vt, "x");
	auto y = graph.add_import(vt, "y");

	jive::phi_builder pb;
	pb.begin_phi(graph.root());

	auto rv1 = pb.add_recvar(ft);
	auto rv2 = pb.add_recvar(ft);
	auto dx = pb.add_dependency(x);
	auto dy = pb.add_dependency(y);

	jlm::lambda_builder lb;
	auto arguments = lb.begin_lambda(pb.region(), {ft, "f", linkage::external_linkage});
	lb.add_dependency(rv1->value());
	lb.add_dependency(dx);
	auto f1 = lb.end_lambda({arguments[0]});

	arguments = lb.begin_lambda(pb.region(), {ft, "g", linkage::external_linkage});
	lb.add_dependency(rv2->value());
	lb.add_dependency(dy);
	auto f2 = lb.end_lambda({arguments[0]});

	rv1->set_value(f1->output(0));
	rv2->set_value(f2->output(0));
	auto phi = pb.end_phi();

	graph.add_export(phi->output(0), "f1");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);
}

static int
verify()
{
	test_root();
	test_gamma();
	test_gamma2();
	test_theta();
	test_nested_theta();
	test_evolving_theta();
	test_lambda();
	test_phi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-dne", verify);
