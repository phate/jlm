/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jive/arch/memorytype.h>
#include <jive/view.h>
#include <jive/vsdg/statemux.h>

#include <jlm/ir/operators/alloca.hpp>
#include <jlm/ir/operators/load.hpp>
#include <jlm/ir/operators/store.hpp>

static inline void
test_load_mux_reduction()
{
	jlm::valuetype vt;
	jlm::ptrtype pt(vt);
	jive::mem::type mt;

	jive::graph graph;
	auto nf = jlm::load_op::normal_form(&graph);
	nf->set_mutable(false);
	nf->set_load_mux_reducible(false);

	auto a = graph.import(pt, "a");
	auto s1 = graph.import(mt, "s1");
	auto s2 = graph.import(mt, "s2");
	auto s3 = graph.import(mt, "s3");

	auto mux = jive::create_state_merge(mt, {s1, s2, s3});
	auto value = jlm::create_load(a, {mux}, 4);

	auto ex = graph.export_port(value, "v");

//	jive::view(graph.root(), stdout);

	nf->set_mutable(true);
	nf->set_load_mux_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto load = ex->origin()->node();
	assert(load && jlm::is_load_op(load->operation()));
	assert(load->ninputs() == 4);
	assert(load->input(1)->origin() == s1);
	assert(load->input(2)->origin() == s2);
	assert(load->input(3)->origin() == s3);
}

static inline void
test_load_alloca_reduction()
{
	jive::mem::type mt;
	jive::bits::type bt(32);

	jive::graph graph;
	auto nf = jlm::load_op::normal_form(&graph);
	nf->set_mutable(false);
	nf->set_load_alloca_reducible(false);

	auto size = graph.import(bt, "v");
	auto state = graph.import(mt, "s");

	auto alloca1 = jlm::create_alloca(bt, size, state, 4);
	auto alloca2 = jlm::create_alloca(bt, size, state, 4);
	auto value = jlm::create_load(alloca1[0], {alloca1[1], alloca2[1]}, 4);

	auto ex = graph.export_port(value, "l");

	jive::view(graph.root(), stdout);

	nf->set_mutable(true);
	nf->set_load_alloca_reducible(true);
	graph.normalize();
	graph.prune();

	jive::view(graph.root(), stdout);

	auto node = ex->origin()->node();
	assert(node && jlm::is_load_op(node->operation()));
	assert(node->ninputs() == 2);
	assert(node->input(1)->origin() == alloca1[1]);
}

static int
test()
{
	test_load_mux_reduction();
	test_load_alloca_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-load", test);
