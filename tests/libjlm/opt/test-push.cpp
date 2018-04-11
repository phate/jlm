/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/arch/addresstype.h>
#include <jive/view.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/theta.h>

#include <jlm/ir/operators/store.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/opt/push.hpp>

static inline void
test_gamma()
{
	jlm::statetype st;
	jlm::valuetype vt;
	jive::ctl::type ct(2);
	jlm::test_op nop({}, {&vt});
	jlm::test_op sop({&vt, &st}, {&st});
	jlm::test_op binop({&vt, &vt}, {&vt});

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");
	auto s = graph.add_import(st, "s");

	auto gamma = jive::gamma_node::create(c, 2);
	auto evx = gamma->add_entryvar(x);
	auto evs = gamma->add_entryvar(s);

	auto null = gamma->subregion(0)->add_simple_node(nop, {})->output(0);
	auto bin = gamma->subregion(0)->add_simple_node(binop, {null, evx->argument(0)})->output(0);
	auto state = gamma->subregion(0)->add_simple_node(sop, {bin, evs->argument(0)})->output(0);

	auto xvs = gamma->add_exitvar({state, evs->argument(1)});

	graph.add_export(gamma->output(0), "x");

//	jive::view(graph.root(), stdout);
	jlm::push(graph);
//	jive::view(graph.root(), stdout);

	assert(graph.root()->nodes.size() == 3);
}

static inline void
test_theta()
{
	jlm::statetype st;
	jlm::valuetype vt;
	jive::ctl::type ct(2);

	jlm::test_op nop({}, {&vt});
	jlm::test_op bop({&vt, &vt}, {&vt});
	jlm::test_op sop({&vt, &st}, {&st});

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto x = graph.add_import(vt, "x");
	auto s = graph.add_import(st, "s");

	auto theta = jive::theta_node::create(graph.root());

	auto lv1 = theta->add_loopvar(c);
	auto lv2 = theta->add_loopvar(x);
	auto lv3 = theta->add_loopvar(x);
	auto lv4 = theta->add_loopvar(s);

	auto o1 = theta->subregion()->add_simple_node(nop, {})->output(0);
	auto o2 = theta->subregion()->add_simple_node(bop, {o1, lv3->argument()})->output(0);
	auto o3 = theta->subregion()->add_simple_node(bop, {lv2->argument(), o2})->output(0);
	auto o4 = theta->subregion()->add_simple_node(sop, {lv3->argument(), lv4->argument()})->output(0);

	lv2->result()->divert_origin(o3);
	lv4->result()->divert_origin(o4);

	theta->set_predicate(lv1->argument());

	graph.add_export(theta->output(0), "c");

//	jive::view(graph.root(), stdout);
	jlm::push(graph);
//	jive::view(graph.root(), stdout);

	assert(graph.root()->nodes.size() == 3);
}

static inline void
test_push_theta_bottom()
{
	jive::memtype mt;
	jlm::valuetype vt;
	jlm::ptrtype pt(vt);
	jive::ctl::type ct(2);

	jive::graph graph;
	auto c = graph.add_import(ct, "c");
	auto a = graph.add_import(pt, "a");
	auto v = graph.add_import(vt, "v");
	auto s = graph.add_import(mt, "s");

	auto theta = jive::theta_node::create(graph.root());

	auto lvc = theta->add_loopvar(c);
	auto lva = theta->add_loopvar(a);
	auto lvv = theta->add_loopvar(v);
	auto lvs = theta->add_loopvar(s);

	auto s1 = jlm::create_store(lva->argument(), lvv->argument(), {lvs->argument()}, 4)[0];

	lvs->result()->divert_origin(s1);
	theta->set_predicate(lvc->argument());

	auto ex = graph.add_export(lvs, "s");

	jive::view(graph, stdout);
	jlm::push_bottom(theta);
	jive::view(graph, stdout);

	auto storenode = ex->origin()->node();
	assert(jlm::is_store_node(storenode));
	assert(storenode->input(0)->origin() == a);
	assert(jive::is_theta_node(storenode->input(1)->origin()->node()));
	assert(jive::is_theta_node(storenode->input(2)->origin()->node()));
}

static int
verify()
{
	test_gamma();
	test_theta();
	test_push_theta_bottom();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-push", verify);
