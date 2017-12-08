/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/types/function/fctlambda.h>
#include <jive/view.h>
#include <jive/rvsdg/control.h>
#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/phi.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/theta.h>

#include <jlm/opt/dne.hpp>

static inline void
test_root()
{
	jive::graph graph;
	graph.import(jlm::valuetype(), "x");
	auto y = graph.import(jlm::valuetype(), "y");
	graph.export_port(y, "z");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(graph.root()->narguments() == 1);
}

static inline void
test_gamma()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op op({&vt}, {&vt});

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");
	auto y = graph.import(vt, "y");

	auto gamma = jive::gamma_node::create(c, 2);
	auto ev1 = gamma->add_entryvar(x);
	auto ev2 = gamma->add_entryvar(y);
	auto ev3 = gamma->add_entryvar(x);

	auto t = gamma->subregion(1)->add_simple_node(op, {ev2->argument(1)})->output(0);

	auto xv1 = gamma->add_exitvar({ev1->argument(0), ev1->argument(1)});
	auto xv2 = gamma->add_exitvar({ev2->argument(0), t});
	auto xv3 = gamma->add_exitvar({ev3->argument(0), ev1->argument(1)});

	graph.export_port(gamma->output(0), "z");
	graph.export_port(gamma->output(2), "w");

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
	jive::ctl::type ct(2);
	jlm::test_op nop({}, {&vt});

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");

	auto gamma = jive::gamma_node::create(c, 2);
	gamma->add_entryvar(x);

	auto n1 = gamma->subregion(0)->add_simple_node(nop, {})->output(0);
	auto n2 = gamma->subregion(1)->add_simple_node(nop, {})->output(0);

	auto xv = gamma->add_exitvar({n1, n2});

	graph.export_port(gamma->output(0), "x");

//	jive::view(graph, stdout);
	jlm::dne(graph);
//	jive::view(graph, stdout);

	assert(graph.root()->narguments() == 1);
}

static inline void
test_theta()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op cop({}, {&ct});
	jlm::test_op op({&vt}, {&vt});

	jive::graph graph;
	auto x = graph.import(vt, "x");
	auto y = graph.import(vt, "y");
	auto z = graph.import(vt, "z");

	auto theta = jive::theta_node::create(graph.root());

	auto lv1 = theta->add_loopvar(x);
	auto lv2 = theta->add_loopvar(y);
	auto lv3 = theta->add_loopvar(z);
	auto lv4 = theta->add_loopvar(y);

	lv1->result()->divert_origin(lv2->argument());
	lv2->result()->divert_origin(lv1->argument());

	auto t = theta->subregion()->add_simple_node(op, {lv3->argument()})->output(0);
	lv3->result()->divert_origin(t);
	lv4->result()->divert_origin(lv2->argument());

	auto c = theta->subregion()->add_simple_node(cop, {})->output(0);
	theta->set_predicate(c);

	graph.export_port(theta->output(0), "a");
	graph.export_port(theta->output(3), "b");

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
	jive::ctl::type ct(2);

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");
	auto y = graph.import(vt, "y");

	auto otheta = jive::theta_node::create(graph.root());

	auto lvo1 = otheta->add_loopvar(c);
	auto lvo2 = otheta->add_loopvar(x);
	auto lvo3 = otheta->add_loopvar(y);

	auto itheta = jive::theta_node::create(otheta->subregion());

	auto lvi1 = itheta->add_loopvar(lvo1->argument());
	auto lvi2 = itheta->add_loopvar(lvo2->argument());
	auto lvi3 = itheta->add_loopvar(lvo3->argument());

	lvi2->result()->divert_origin(lvi3->argument());

	itheta->set_predicate(lvi1->argument());

	lvo2->result()->divert_origin(itheta->output(1));
	lvo3->result()->divert_origin(itheta->output(1));

	otheta->set_predicate(lvo1->argument());

	graph.export_port(otheta->output(2), "y");

//	jive::view(graph, stdout);
	jlm::dne(graph);
//	jive::view(graph, stdout);

	assert(otheta->noutputs() == 3);
}

static inline void
test_evolving_theta()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op bop({&vt, &vt}, {&vt});

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x1 = graph.import(vt, "x1");
	auto x2 = graph.import(vt, "x2");
	auto x3 = graph.import(vt, "x3");
	auto x4 = graph.import(vt, "x4");

	auto theta = jive::theta_node::create(graph.root());

	auto lv0 = theta->add_loopvar(c);
	auto lv1 = theta->add_loopvar(x1);
	auto lv2 = theta->add_loopvar(x2);
	auto lv3 = theta->add_loopvar(x3);
	auto lv4 = theta->add_loopvar(x4);

	lv1->result()->divert_origin(lv2->argument());
	lv2->result()->divert_origin(lv3->argument());
	lv3->result()->divert_origin(lv4->argument());

	theta->set_predicate(lv0->argument());

	graph.export_port(lv1->output(), "x1");

//	jive::view(graph, stdout);
	jlm::dne(graph);
//	jive::view(graph, stdout);

	assert(theta->noutputs() == 5);
}

static inline void
test_lambda()
{
	jlm::valuetype vt;
	jlm::test_op op({&vt, &vt}, {&vt});

	jive::graph graph;
	auto x = graph.import(vt, "x");

	jive::lambda_builder lb;
	auto arguments = lb.begin_lambda(graph.root(), {{&vt}, {&vt}});

	auto d = lb.add_dependency(x);
	lb.subregion()->add_simple_node(op, {arguments[0], d});

	auto lambda = lb.end_lambda({arguments[0]});

	graph.export_port(lambda->output(0), "f");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(lambda->subregion()->nodes.size() == 0);
	assert(graph.root()->narguments() == 0);
}

static inline void
test_phi()
{
	jlm::valuetype vt;
	jive::fct::type ft({&vt}, {&vt});

	jive::graph graph;
	auto x = graph.import(vt, "x");
	auto y = graph.import(vt, "y");

	jive::phi_builder pb;
	pb.begin_phi(graph.root());

	auto rv1 = pb.add_recvar(ft);
	auto rv2 = pb.add_recvar(ft);
	auto dx = pb.add_dependency(x);
	auto dy = pb.add_dependency(y);

	jive::lambda_builder lb;
	auto arguments = lb.begin_lambda(pb.region(), ft);
	lb.add_dependency(rv1->value());
	lb.add_dependency(dx);
	auto f1 = lb.end_lambda({arguments[0]});

	arguments = lb.begin_lambda(pb.region(), ft);
	lb.add_dependency(rv2->value());
	lb.add_dependency(dy);
	auto f2 = lb.end_lambda({arguments[0]});

	rv1->set_value(f1->output(0));
	rv2->set_value(f2->output(0));
	auto phi = pb.end_phi();

	graph.export_port(phi->output(0), "f1");

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

JLM_UNIT_TEST_REGISTER("libjlm/test-dne", verify);
