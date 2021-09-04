/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jive/arch/addresstype.hpp>
#include <jive/view.hpp>
#include <jive/rvsdg/statemux.hpp>

#include <jlm/ir/operators/alloca.hpp>
#include <jlm/ir/operators/load.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/ir/operators/store.hpp>

static inline void
test_load_mux_reduction()
{
	using namespace jlm;

	jlm::valuetype vt;
	jlm::ptrtype pt(vt);
	jive::memtype mt;

	jive::graph graph;
	auto nf = jlm::load_op::normal_form(&graph);
	nf->set_mutable(false);
	nf->set_load_mux_reducible(false);

	auto a = graph.add_import({pt, "a"});
	auto s1 = graph.add_import({mt, "s1"});
	auto s2 = graph.add_import({mt, "s2"});
	auto s3 = graph.add_import({mt, "s3"});

	auto mux = MemStateMergeOperator::Create({s1, s2, s3});
	auto value = load_op::create(a, {mux}, 4)[0];

	auto ex = graph.add_export(value, {value->type(), "v"});

//	jive::view(graph.root(), stdout);

	nf->set_mutable(true);
	nf->set_load_mux_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto load = jive::node_output::node(ex->origin());
	assert(jive::is<jlm::load_op>(load));
	assert(load->ninputs() == 4);
	assert(load->input(1)->origin() == s1);
	assert(load->input(2)->origin() == s2);
	assert(load->input(3)->origin() == s3);
}

static inline void
test_load_alloca_reduction()
{
	using namespace jlm;

	jive::memtype mt;
	jive::bittype bt(32);

	jive::graph graph;
	auto nf = jlm::load_op::normal_form(&graph);
	nf->set_mutable(false);
	nf->set_load_alloca_reducible(false);

	auto size = graph.add_import({bt, "v"});

	auto alloca1 = alloca_op::create(bt, size, 4);
	auto alloca2 = alloca_op::create(bt, size, 4);
	auto mux = jive::create_state_mux(mt, {alloca1[1]}, 1);
	auto value = load_op::create(alloca1[0], {alloca1[1], alloca2[1], mux[0]}, 4)[0];

	auto ex = graph.add_export(value, {value->type(), "l"});

//	jive::view(graph.root(), stdout);

	nf->set_mutable(true);
	nf->set_load_alloca_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto node = jive::node_output::node(ex->origin());
	assert(jive::is<load_op>(node));
	assert(node->ninputs() == 3);
	assert(node->input(1)->origin() == alloca1[1]);
	assert(node->input(2)->origin() == mux[0]);
}

static inline void
test_multiple_origin_reduction()
{
	using namespace jlm;

	jive::memtype mt;
	jlm::valuetype vt;
	jlm::ptrtype pt(vt);

	jive::graph graph;
	auto nf = jlm::load_op::normal_form(&graph);
	nf->set_mutable(false);
	nf->set_multiple_origin_reducible(false);

	auto a = graph.add_import({pt, "a"});
	auto s = graph.add_import({mt, "s"});

	auto load = load_op::create(a, {s, s, s, s}, 4)[0];

	auto ex = graph.add_export(load, {load->type(), "l"});

//	jive::view(graph.root(), stdout);

	nf->set_mutable(true);
	nf->set_multiple_origin_reducible(true);
	graph.normalize();

//	jive::view(graph.root(), stdout);

	auto node = jive::node_output::node(ex->origin());
	assert(is<load_op>(node));
	assert(node->ninputs() == 2);
}

static inline void
test_load_store_state_reduction()
{
	using namespace jlm;

	jive::bittype bt(32);

	jive::graph graph;
	auto nf = jlm::load_op::normal_form(&graph);
	nf->set_mutable(false);
	nf->set_load_store_state_reducible(false);

	auto size = graph.add_import({bt, "v"});

	auto alloca1 = alloca_op::create(bt, size, 4);
	auto alloca2 = alloca_op::create(bt, size, 4);
	auto store1 = store_op::create(alloca1[0], size, {alloca1[1]}, 4);
	auto store2 = store_op::create(alloca2[0], size, {alloca2[1]}, 4);

	auto value1 = load_op::create(alloca1[0], {store1[0], store2[0]}, 4)[0];
	auto value2 = load_op::create(alloca1[0], {store1[0]}, 8)[0];

	auto ex1 = graph.add_export(value1, {value1->type(), "l1"});
	auto ex2 = graph.add_export(value2, {value2->type(), "l2"});

//	jive::view(graph.root(), stdout);

	nf->set_mutable(true);
	nf->set_load_store_state_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto node = jive::node_output::node(ex1->origin());
	assert(is<load_op>(node));
	assert(node->ninputs() == 2);

	node = jive::node_output::node(ex2->origin());
	assert(is<load_op>(node));
	assert(node->ninputs() == 2);
}

static inline void
test_load_store_alloca_reduction()
{
	using namespace jlm;

	jive::memtype mt;
	jive::bittype bt(32);

	jive::graph graph;
	auto nf = jlm::load_op::normal_form(&graph);
	nf->set_mutable(false);
	nf->set_load_store_alloca_reducible(false);

	auto size = graph.add_import({bt, "v"});

	auto alloca = alloca_op::create(bt, size, 4);
	auto store = store_op::create(alloca[0], size, {alloca[1]}, 4);
	auto load = load_op::create(alloca[0], store, 4);

	auto value = graph.add_export(load[0], {load[0]->type(), "l"});
	auto rstate = graph.add_export(load[1], {mt, "s"});

//	jive::view(graph.root(), stdout);

	nf->set_mutable(true);
	nf->set_load_store_alloca_reducible(true);
	graph.normalize();

//	jive::view(graph.root(), stdout);

	assert(value->origin() == graph.root()->argument(0));
	assert(rstate->origin() == alloca[1]);
}

static inline void
test_load_store_reduction()
{
	using namespace jlm;

	valuetype vt;
	ptrtype pt(vt);
	jive::memtype mt;

	jive::graph graph;
	auto nf = load_op::normal_form(&graph);
	nf->set_load_store_reducible(true);

	auto a = graph.add_import({pt, "address"});
	auto v = graph.add_import({vt, "value"});
	auto s = graph.add_import({mt, "state"});

	auto s1 = store_op::create(a, v, {s}, 4)[0];
	auto load = load_op::create(a, {s1}, 4);

	auto x1 = graph.add_export(load[0], {load[0]->type(), "value"});
	auto x2 = graph.add_export(load[1], {load[1]->type(), "state"});

	jive::view(graph.root(), stdout);

	assert(graph.root()->nnodes() == 1);
	assert(x1->origin() == v);
	assert(x2->origin() == s1);
}

static void
test_load_load_reduction()
{
	using namespace jlm;

	valuetype vt;
	ptrtype pt(vt);
	jive::memtype mt;

	jive::graph graph;
	auto nf = load_op::normal_form(&graph);
	nf->set_mutable(false);

	auto a1 = graph.add_import({pt, "a1"});
	auto a2 = graph.add_import({pt, "a2"});
	auto a3 = graph.add_import({pt, "a3"});
	auto a4 = graph.add_import({pt, "a4"});
	auto v1 = graph.add_import({vt, "v1"});
	auto s1 = graph.add_import({mt, "s1"});
	auto s2 = graph.add_import({mt, "s2"});

	auto st1 = store_op::create(a1, v1, {s1}, 4);
	auto ld1 = load_op::create(a2, {s1}, 4);
	auto ld2 = load_op::create(a3, {s2}, 4);

	auto ld3 = load_op::create(a4, {st1[0], ld1[1], ld2[1]}, 4);

	auto x1 = graph.add_export(ld3[1], {mt, "s"});
	auto x2 = graph.add_export(ld3[2], {mt, "s"});
	auto x3 = graph.add_export(ld3[3], {mt, "s"});

	jive::view(graph.root(), stdout);

	nf->set_mutable(true);
	nf->set_load_load_state_reducible(true);
	graph.normalize();
	graph.prune();

	jive::view(graph.root(), stdout);

	assert(graph.root()->nnodes() == 6);

	auto ld = jive::node_output::node(x1->origin());
	assert(is<load_op>(ld));

	auto mx1 = jive::node_output::node(x2->origin());
	assert(is<MemStateMergeOperator>(mx1) && mx1->ninputs() == 2);
	assert(mx1->input(0)->origin() == ld1[1] || mx1->input(0)->origin() == ld->output(2));
	assert(mx1->input(1)->origin() == ld1[1] || mx1->input(1)->origin() == ld->output(2));

	auto mx2 = jive::node_output::node(x3->origin());
	assert(is<MemStateMergeOperator>(mx2) && mx2->ninputs() == 2);
	assert(mx2->input(0)->origin() == ld2[1] || mx2->input(0)->origin() == ld->output(3));
	assert(mx2->input(1)->origin() == ld2[1] || mx2->input(1)->origin() == ld->output(3));
}

static int
test()
{
	test_load_mux_reduction();
	test_load_alloca_reduction();
	test_multiple_origin_reduction();
	test_load_store_state_reduction();
	test_load_store_alloca_reduction();
	test_load_store_reduction();
	test_load_load_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-load", test)
