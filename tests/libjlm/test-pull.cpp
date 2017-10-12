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

#include <jlm/opt/pull.hpp>

static inline void
test_pullin_top()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op uop({&vt}, {&vt});
	jlm::test_op bop({&vt, &vt}, {&vt});
	jlm::test_op cop({&ct, &vt}, {&ct});

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");

	auto n1 = graph.root()->add_simple_node(uop, {x});
	auto n2 = graph.root()->add_simple_node(uop, {x});
	auto n3 = graph.root()->add_simple_node(uop, {n2->output(0)});
	auto n4 = graph.root()->add_simple_node(cop, {c, n1->output(0)});
	auto n5 = graph.root()->add_simple_node(bop, {n1->output(0), n3->output(0)});

	jive::gamma_builder gb;
	gb.begin_gamma(n4->output(0));

	gb.add_entryvar(n4->output(0));
	auto ev = gb.add_entryvar(n5->output(0));
	auto xv = gb.add_exitvar({ev->argument(0), ev->argument(1)});

	auto gamma = gb.end_gamma();

	graph.export_port(gamma->node()->output(0), "x");
	graph.export_port(n2->output(0), "y");

//	jive::view(graph, stdout);
	jlm::pullin_top(*gamma);
//	jive::view(graph, stdout);

	assert(gamma->node()->subregion(0)->nnodes() == 2);
	assert(gamma->node()->subregion(1)->nnodes() == 2);
}

static inline void
test_pullin_bottom()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op bop({&vt, &vt}, {&vt});

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");

	jive::gamma_builder gb;
	gb.begin_gamma(c);

	auto ev = gb.add_entryvar(x);
	auto xv = gb.add_exitvar({ev->argument(0), ev->argument(1)});

	auto gamma = gb.end_gamma();

	auto node = gamma->node();
	auto b1 = graph.root()->add_simple_node(bop, {node->output(0), x});
	auto b2 = graph.root()->add_simple_node(bop, {node->output(0), b1->output(0)});

	auto xp = graph.export_port(b2->output(0), "x");

//	jive::view(graph, stdout);
	jlm::pullin_bottom(*gamma);
//	jive::view(graph, stdout);

	assert(xp->origin()->node() == node);
	assert(node->subregion(0)->nnodes() == 2);
	assert(node->subregion(1)->nnodes() == 2);
}

static int
verify()
{
	test_pullin_top();
	test_pullin_bottom();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-pull", verify);
