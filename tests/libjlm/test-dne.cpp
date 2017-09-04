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

	jive::gamma_builder gb;
	gb.begin(c);
	auto ev1 = gb.add_entryvar(x);
	auto ev2 = gb.add_entryvar(y);

	auto t = gb.region(1)->add_simple_node(op, {ev2->argument(1)})->output(0);

	auto xv1 = gb.add_exitvar({ev1->argument(0), ev1->argument(1)});
	auto xv2 = gb.add_exitvar({ev2->argument(0), t});

	auto gamma = gb.end();

	graph.export_port(gamma->output(0), "z");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(gamma->noutputs() == 1);
	assert(gamma->subregion(1)->nodes.size() == 0);
	assert(gamma->ninputs() == 2);
	assert(graph.root()->narguments() == 2);
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

	jive::theta_builder tb;
	tb.begin(graph.root());

	auto lv1 = tb.add_loopvar(x);
	auto lv2 = tb.add_loopvar(y);
	auto lv3 = tb.add_loopvar(z);

	auto tmp = lv1->value();
	lv1->set_value(lv2->value());
	lv2->set_value(tmp);

	auto t = tb.region()->add_simple_node(op, {lv3->value()})->output(0);
	lv3->set_value(t);

	auto c = tb.region()->add_simple_node(cop, {})->output(0);
	auto theta = tb.end(c);

	graph.export_port(theta->output(0), "a");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(theta->noutputs() == 2);
	assert(theta->subregion(0)->nodes.size() == 1);
	assert(graph.root()->narguments() == 2);
}

static inline void
test_lambda()
{
	jlm::valuetype vt;
	jlm::test_op op({&vt, &vt}, {&vt});

	jive::graph graph;
	auto x = graph.import(vt, "x");

	jive::lambda_builder lb;
	lb.begin(graph.root(), {{&vt}, {&vt}});

	auto d = lb.add_dependency(x);
	lb.region()->add_simple_node(op, {lb.region()->argument(0), d});

	auto lambda = lb.end({lb.region()->argument(0)});

	graph.export_port(lambda->output(0), "f");

//	jive::view(graph.root(), stdout);
	jlm::dne(graph);
//	jive::view(graph.root(), stdout);

	assert(lambda->subregion(0)->nodes.size() == 0);
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
	pb.begin(graph.root());

	auto rv1 = pb.add_recvar(ft);
	auto rv2 = pb.add_recvar(ft);
	auto dx = pb.add_dependency(x);
	auto dy = pb.add_dependency(y);

	jive::lambda_builder lb;
	lb.begin(pb.region(), ft);
	lb.add_dependency(rv1->value());
	lb.add_dependency(dx);
	auto f1 = lb.end({lb.region()->argument(0)});

	lb.begin(pb.region(), ft);
	lb.add_dependency(rv2->value());
	lb.add_dependency(dy);
	auto f2 = lb.end({lb.region()->argument(0)});

	rv1->set_value(f1->output(0));
	rv2->set_value(f2->output(0));
	auto phi = pb.end();

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
	test_theta();
	test_lambda();
	test_phi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-dne", verify);
