/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jive/arch/memorytype.h>
#include <jive/types/bitstring/type.h>
#include <jive/view.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/statemux.h>

#include <jlm/ir/operators/alloca.hpp>
#include <jlm/ir/operators/store.hpp>
#include <jlm/ir/types.hpp>

static inline void
test_store_mux_reduction()
{
	jlm::valuetype vt;
	jlm::ptrtype pt(vt);
	jive::mem::type mt;

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::store_op));
	auto snf = static_cast<jlm::store_normal_form*>(nf);
	snf->set_mutable(false);
	snf->set_store_mux_reducible(false);

	auto a = graph.import(pt, "a");
	auto v = graph.import(vt, "v");
	auto s1 = graph.import(mt, "s1");
	auto s2 = graph.import(mt, "s2");
	auto s3 = graph.import(mt, "s3");

	auto mux = jive::create_state_merge(mt, {s1, s2, s3});
	auto state = jlm::create_store(a, v, {mux}, 4);

	auto ex = graph.export_port(state[0], "s");

//	jive::view(graph.root(), stdout);

	snf->set_mutable(true);
	snf->set_store_mux_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto muxnode= ex->origin()->node();
	assert(is_mux_op(muxnode->operation()));
	assert(muxnode->ninputs() == 3);
	assert(jlm::is_store_op(muxnode->input(0)->origin()->node()->operation()));
	assert(jlm::is_store_op(muxnode->input(1)->origin()->node()->operation()));
	assert(jlm::is_store_op(muxnode->input(2)->origin()->node()->operation()));
}

static inline void
test_multiple_origin_reduction()
{
	jlm::valuetype vt;
	jlm::ptrtype pt(vt);
	jive::mem::type mt;

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::store_op));
	auto snf = static_cast<jlm::store_normal_form*>(nf);
	snf->set_mutable(false);
	snf->set_multiple_origin_reducible(false);

	auto a = graph.import(pt, "a");
	auto v = graph.import(vt, "v");
	auto s = graph.import(mt, "s");

	auto states = jlm::create_store(a, v, {s, s, s, s}, 4);

	auto ex = graph.export_port(states[0], "s");

//	jive::view(graph.root(), stdout);

	snf->set_mutable(true);
	snf->set_multiple_origin_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto node = ex->origin()->node();
	assert(jlm::is_store_op(node->operation()) && node->ninputs() == 3);
}

static inline void
test_store_alloca_reduction()
{
	jlm::valuetype vt;
	jive::mem::type mt;
	jive::bits::type bt(32);

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::store_op));
	auto snf = static_cast<jlm::store_normal_form*>(nf);
	snf->set_mutable(false);
	snf->set_store_alloca_reducible(false);

	auto size = graph.import(bt, "size");
	auto value = graph.import(vt, "value");
	auto s = graph.import(mt, "s");

	auto alloca1 = jlm::create_alloca(vt, size, s, 4);
	auto alloca2 = jlm::create_alloca(vt, size, s, 4);
	auto states1 = jlm::create_store(alloca1[0], value, {alloca1[1], alloca2[1], s}, 4);
	auto states2 = jlm::create_store(alloca2[0], value, states1, 4);

	graph.export_port(states2[0], "s1");
	graph.export_port(states2[1], "s2");
	graph.export_port(states2[2], "s3");

	jive::view(graph.root(), stdout);

	snf->set_mutable(true);
	snf->set_store_alloca_reducible(true);
	graph.normalize();
	graph.prune();

	jive::view(graph.root(), stdout);

	bool has_import = false;
	for (size_t n = 0; n < graph.root()->nresults(); n++) {
		if (graph.root()->result(n)->origin() == s)
			has_import = true;
	}
	assert(has_import);
}

static int
test()
{
	test_store_mux_reduction();
	test_store_alloca_reduction();
	test_multiple_origin_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-store", test);
