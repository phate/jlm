/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jive/arch/addresstype.h>
#include <jive/types/bitstring/type.h>
#include <jive/view.h>
#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/statemux.h>

#include <jlm/jlm/ir/operators/alloca.hpp>
#include <jlm/jlm/ir/operators/store.hpp>
#include <jlm/jlm/ir/types.hpp>

static inline void
test_store_mux_reduction()
{
	jlm::valuetype vt;
	jlm::ptrtype pt(vt);
	jive::memtype mt;

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::store_op));
	auto snf = static_cast<jlm::store_normal_form*>(nf);
	snf->set_mutable(false);
	snf->set_store_mux_reducible(false);

	auto a = graph.add_import(pt, "a");
	auto v = graph.add_import(vt, "v");
	auto s1 = graph.add_import(mt, "s1");
	auto s2 = graph.add_import(mt, "s2");
	auto s3 = graph.add_import(mt, "s3");

	auto mux = jive::create_state_merge(mt, {s1, s2, s3});
	auto state = jlm::create_store(a, v, {mux}, 4);

	auto ex = graph.add_export(state[0], "s");

//	jive::view(graph.root(), stdout);

	snf->set_mutable(true);
	snf->set_store_mux_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto muxnode= ex->origin()->node();
	assert(jive::is<jive::mux_op>(muxnode->operation()));
	assert(muxnode->ninputs() == 3);
	assert(jive::is<jlm::store_op>(muxnode->input(0)->origin()->node()->operation()));
	assert(jive::is<jlm::store_op>(muxnode->input(1)->origin()->node()->operation()));
	assert(jive::is<jlm::store_op>(muxnode->input(2)->origin()->node()->operation()));
}

static inline void
test_multiple_origin_reduction()
{
	jlm::valuetype vt;
	jlm::ptrtype pt(vt);
	jive::memtype mt;

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::store_op));
	auto snf = static_cast<jlm::store_normal_form*>(nf);
	snf->set_mutable(false);
	snf->set_multiple_origin_reducible(false);

	auto a = graph.add_import(pt, "a");
	auto v = graph.add_import(vt, "v");
	auto s = graph.add_import(mt, "s");

	auto states = jlm::create_store(a, v, {s, s, s, s}, 4);

	auto ex = graph.add_export(states[0], "s");

//	jive::view(graph.root(), stdout);

	snf->set_mutable(true);
	snf->set_multiple_origin_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto node = ex->origin()->node();
	assert(jive::is<jlm::store_op>(node->operation()) && node->ninputs() == 3);
}

static inline void
test_store_alloca_reduction()
{
	jlm::valuetype vt;
	jive::memtype mt;
	jive::bittype bt(32);

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jlm::store_op));
	auto snf = static_cast<jlm::store_normal_form*>(nf);
	snf->set_mutable(false);
	snf->set_store_alloca_reducible(false);

	auto size = graph.add_import(bt, "size");
	auto value = graph.add_import(vt, "value");
	auto s = graph.add_import(mt, "s");

	auto alloca1 = jlm::create_alloca(vt, size, s, 4);
	auto alloca2 = jlm::create_alloca(vt, size, s, 4);
	auto states1 = jlm::create_store(alloca1[0], value, {alloca1[1], alloca2[1], s}, 4);
	auto states2 = jlm::create_store(alloca2[0], value, states1, 4);

	graph.add_export(states2[0], "s1");
	graph.add_export(states2[1], "s2");
	graph.add_export(states2[2], "s3");

//	jive::view(graph.root(), stdout);

	snf->set_mutable(true);
	snf->set_store_alloca_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	bool has_add_import = false;
	for (size_t n = 0; n < graph.root()->nresults(); n++) {
		if (graph.root()->result(n)->origin() == s)
			has_add_import = true;
	}
	assert(has_add_import);
}

static inline void
test_store_store_reduction()
{
	using namespace jlm;

	valuetype vt;
	jlm::ptrtype pt(vt);
	jive::memtype mt;

	jive::graph graph;
	auto a = graph.add_import(pt, "address");
	auto v1 = graph.add_import(vt, "value");
	auto v2 = graph.add_import(vt, "value");
	auto s = graph.add_import(mt, "state");

	auto s1 = jlm::create_store(a, v1, {s}, 4)[0];
	auto s2 = jlm::create_store(a, v2, {s1}, 4)[0];

	auto ex = graph.add_export(s2, "state");

	jive::view(graph.root(), stdout);

	auto nf = store_op::normal_form(&graph);
	nf->set_store_store_reducible(true);
	graph.normalize();
	graph.prune();

	jive::view(graph.root(), stdout);

	assert(graph.root()->nnodes() == 1);
	assert(ex->origin()->node()->input(1)->origin() == v2);
}

static int
test()
{
	test_store_mux_reduction();
	test_store_alloca_reduction();
	test_multiple_origin_reduction();
	test_store_store_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-store", test)
