/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

static int
test_main()
{
	using namespace jive;

	jlm::valuetype t;

	jive::graph graph;
	auto i = graph.add_import({t, "i"});

	auto o1 = jlm::test_op::create(graph.root(), {}, {&t})->output(0);
	auto o2 = jlm::test_op::create(graph.root(), {i}, {&t})->output(0);

	auto e1 = graph.add_export(o1, {o1->type(), "o1"});
	auto e2 = graph.add_export(o2, {o2->type(), "o2"});

	auto nf = dynamic_cast<jive::simple_normal_form*>(graph.node_normal_form(
		typeid(jlm::test_op)));
	nf->set_mutable(false);

	auto o3 = jlm::create_testop(graph.root(), {}, {&t})[0];
	auto o4 = jlm::create_testop(graph.root(), {i}, {&t})[0];

	auto e3 = graph.add_export(o3, {o3->type(), "o3"});
	auto e4 = graph.add_export(o4, {o4->type(), "o4"});

	nf->set_mutable(true);
	graph.normalize();
	assert(e1->origin() == e3->origin());
	assert(e2->origin() == e4->origin());

	auto o5 = jlm::create_testop(graph.root(), {}, {&t})[0];
	assert(o5 == e1->origin());

	auto o6 = jlm::create_testop(graph.root(), {i}, {&t})[0];
	assert(o6 == e2->origin());

	nf->set_cse(false);

	auto o7 = jlm::create_testop(graph.root(), {}, {&t})[0];
	assert(o7 != e1->origin());

	graph.normalize();
	assert(o7 != e1->origin());

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-cse", test_main)
