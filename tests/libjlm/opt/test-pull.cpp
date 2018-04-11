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
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");

	auto n1 = graph.root()->add_simple_node(uop, {x});
	auto n2 = graph.root()->add_simple_node(uop, {x});
	auto n3 = graph.root()->add_simple_node(uop, {n2->output(0)});
	auto n4 = graph.root()->add_simple_node(cop, {c, n1->output(0)});
	auto n5 = graph.root()->add_simple_node(bop, {n1->output(0), n3->output(0)});

	auto gamma = jive::gamma_node::create(n4->output(0), 2);

	gamma->add_entryvar(n4->output(0));
	auto ev = gamma->add_entryvar(n5->output(0));
	auto xv = gamma->add_exitvar({ev->argument(0), ev->argument(1)});

	graph.add_export(gamma->output(0), "x");
	graph.add_export(n2->output(0), "y");

//	jive::view(graph, stdout);
	jlm::pullin_top(gamma);
//	jive::view(graph, stdout);

	assert(gamma->subregion(0)->nnodes() == 2);
	assert(gamma->subregion(1)->nnodes() == 2);
}

static inline void
test_pullin_bottom()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op bop({&vt, &vt}, {&vt});

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");

	auto gamma = jive::gamma_node::create(c, 2);

	auto ev = gamma->add_entryvar(x);
	auto xv = gamma->add_exitvar({ev->argument(0), ev->argument(1)});

	auto b1 = graph.root()->add_simple_node(bop, {gamma->output(0), x});
	auto b2 = graph.root()->add_simple_node(bop, {gamma->output(0), b1->output(0)});

	auto xp = graph.add_export(b2->output(0), "x");

//	jive::view(graph, stdout);
	jlm::pullin_bottom(gamma);
//	jive::view(graph, stdout);

	assert(xp->origin()->node() == gamma);
	assert(gamma->subregion(0)->nnodes() == 2);
	assert(gamma->subregion(1)->nnodes() == 2);
}

static int
verify()
{
	test_pullin_top();
	test_pullin_bottom();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-pull", verify);
