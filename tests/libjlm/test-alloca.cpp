/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <jlm/ir/operators/alloca.hpp>

static inline void
test_alloca_alloca_reduction()
{
	jive::mem::type mt;
	jive::bits::type bt(32);

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::alloca_op));
	auto anf = static_cast<jlm::alloca_normal_form*>(nf);
	anf->set_mutable(false);
	anf->set_alloca_alloca_reducible(false);

	auto size = graph.import(bt, "size");
	auto state = graph.import(mt, "state");

	auto outputs = jlm::create_alloca(bt, size, state, 4);
	outputs = jlm::create_alloca(bt, size, outputs[1], 4);

	graph.export_port(outputs[0], "address");
	auto exs = graph.export_port(outputs[1], "state");

	jive::view(graph.root(), stdout);

	anf->set_mutable(true);
	anf->set_alloca_alloca_reducible(true);
	graph.normalize();

	jive::view(graph.root(), stdout);

	auto mux = exs->origin()->node();
	assert(dynamic_cast<const jive::state::mux_op*>(&mux->operation()));
	auto alloca1 = mux->input(0)->origin()->node();
	auto alloca2 = mux->input(1)->origin()->node();
	assert(dynamic_cast<const jlm::alloca_op*>(&alloca1->operation()));
	assert(dynamic_cast<const jlm::alloca_op*>(&alloca2->operation()));
	assert(alloca1 != alloca2);
	assert(alloca1->input(1)->origin() == alloca2->input(1)->origin());
}

static int
test()
{
	test_alloca_alloca_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-alloca", test);
