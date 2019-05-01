/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jive/view.h>
#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/statemux.h>

#include <jlm/ir/operators/alloca.hpp>

static inline void
test_alloca_alloca_reduction()
{
	jive::memtype mt;
	jive::bittype bt(32);

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::alloca_op));
	auto anf = static_cast<jlm::alloca_normal_form*>(nf);
	anf->set_mutable(false);
	anf->set_alloca_alloca_reducible(false);

	auto size = graph.add_import({bt, "size"});
	auto state = graph.add_import({mt, "state"});

	auto outputs = jlm::create_alloca(bt, size, state, 4);
	outputs = jlm::create_alloca(bt, size, outputs[1], 4);

	graph.add_export(outputs[0], {outputs[0]->type(), "address"});
	auto exs = graph.add_export(outputs[1], {outputs[1]->type(), "state"});

//	jive::view(graph.root(), stdout);

	anf->set_mutable(true);
	anf->set_alloca_alloca_reducible(true);
	graph.normalize();

//	jive::view(graph.root(), stdout);

	auto mux = exs->origin()->node();
	assert(dynamic_cast<const jive::mux_op*>(&mux->operation()));
	auto alloca1 = mux->input(0)->origin()->node();
	auto alloca2 = mux->input(1)->origin()->node();
	assert(dynamic_cast<const jlm::alloca_op*>(&alloca1->operation()));
	assert(dynamic_cast<const jlm::alloca_op*>(&alloca2->operation()));
	assert(alloca1 != alloca2);
	assert(alloca1->input(1)->origin() == alloca2->input(1)->origin());
}

static inline void
test_alloca_mux_reduction()
{
	jive::memtype mt;
	jive::bittype bt(32);

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::alloca_op));
	auto anf = static_cast<jlm::alloca_normal_form*>(nf);
	anf->set_mutable(false);
	anf->set_alloca_mux_reducible(false);

	auto size = graph.add_import({bt, "size"});
	auto state = graph.add_import({mt, "state"});

	auto alloc1 = jlm::create_alloca(bt, size, state, 4);
	auto alloc2 = jlm::create_alloca(bt, size, state, 4);

	auto s = jive::create_state_merge(mt, {alloc1[1], alloc2[1]});

	auto alloc3 = jlm::create_alloca(bt, size, s, 4);

	graph.add_export(alloc3[0], {alloc3[0]->type(), "address"});
	auto exs = graph.add_export(alloc3[1], {alloc3[1]->type(), "state"});

	jive::view(graph.root(), stdout);

	anf->set_mutable(true);
	anf->set_alloca_mux_reducible(true);
	graph.normalize();
	graph.prune();

	jive::view(graph.root(), stdout);

	auto mux = exs->origin()->node();
	assert(dynamic_cast<const jive::mux_op*>(&mux->operation()));
	auto n1 = mux->input(0)->origin()->node();
	auto n2 = mux->input(1)->origin()->node();
	assert(jlm::is_alloca_op(n1->operation()) || jlm::is_alloca_op(n2->operation()));
	assert(dynamic_cast<const jive::mux_op*>(&n1->operation())
		|| dynamic_cast<const jive::mux_op*>(&n2->operation()));
}

static int
test()
{
	test_alloca_alloca_reduction();
	test_alloca_mux_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-alloca", test)
