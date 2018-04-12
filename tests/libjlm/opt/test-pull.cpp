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
	jive::ctltype ct(2);
	jlm::test_op uop({&vt}, {&vt});
	jlm::test_op bop({&vt, &vt}, {&vt});
	jlm::test_op cop({&ct, &vt}, {&ct});

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");

	auto n1 = jlm::create_testop(graph.root(), {x}, {&vt})[0];
	auto n2 = jlm::create_testop(graph.root(), {x}, {&vt})[0];
	auto n3 = jlm::create_testop(graph.root(), {n2}, {&vt})[0];
	auto n4 = jlm::create_testop(graph.root(), {c, n1}, {&ct})[0];
	auto n5 = jlm::create_testop(graph.root(), {n1, n3}, {&vt})[0];

	auto gamma = jive::gamma_node::create(n4, 2);

	gamma->add_entryvar(n4);
	auto ev = gamma->add_entryvar(n5);
	gamma->add_exitvar({ev->argument(0), ev->argument(1)});

	graph.add_export(gamma->output(0), "x");
	graph.add_export(n2, "y");

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
	jive::ctltype ct(2);

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");

	auto gamma = jive::gamma_node::create(c, 2);

	auto ev = gamma->add_entryvar(x);
	gamma->add_exitvar({ev->argument(0), ev->argument(1)});

	auto b1 = jlm::create_testop(graph.root(), {gamma->output(0), x}, {&vt})[0];
	auto b2 = jlm::create_testop(graph.root(), {gamma->output(0), b1}, {&vt})[0];

	auto xp = graph.add_export(b2, "x");

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
