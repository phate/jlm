/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <assert.h>

#include <jlm/rvsdg/statemux.hpp>
#include <jlm/rvsdg/view.hpp>


static void
test_mux_mux_reduction()
{
	using namespace jive;

	jlm::statetype st;

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jive::mux_op));
	auto mnf = static_cast<jive::mux_normal_form*>(nf);
	mnf->set_mutable(false);
	mnf->set_mux_mux_reducible(false);

	auto x = graph.add_import({st, "x"});
	auto y = graph.add_import({st, "y"});
	auto z = graph.add_import({st, "z"});

	auto mux1 = jive::create_state_merge(st, {x, y});
	auto mux2 = jive::create_state_split(st, z, 2);
	auto mux3 = jive::create_state_merge(st, {mux1, mux2[0], mux2[1], z});

	auto ex = graph.add_export(mux3, {mux3->type(), "m"});

//	jive::view(graph.root(), stdout);

	mnf->set_mutable(true);
	mnf->set_mux_mux_reducible(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph.root(), stdout);

	auto node = node_output::node(ex->origin());
	assert(node->ninputs() == 4);
	assert(node->input(0)->origin() == x);
	assert(node->input(1)->origin() == y);
	assert(node->input(2)->origin() == z);
	assert(node->input(3)->origin() == z);
}

static void
test_multiple_origin_reduction()
{
	using namespace jive;

	jlm::statetype st;

	jive::graph graph;
	auto nf = graph.node_normal_form(typeid(jive::mux_op));
	auto mnf = static_cast<jive::mux_normal_form*>(nf);
	mnf->set_mutable(false);
	mnf->set_multiple_origin_reducible(false);

	auto x = graph.add_import({st, "x"});
	auto mux1 = jive::create_state_merge(st, {x, x});
	auto ex = graph.add_export(mux1, {mux1->type(), "m"});

	view(graph.root(), stdout);

	mnf->set_mutable(true);
	mnf->set_multiple_origin_reducible(true);
	graph.normalize();
	graph.prune();

	view(graph.root(), stdout);

	assert(node_output::node(ex->origin())->ninputs() == 1);
}

static int
test_main(void)
{
	test_mux_mux_reduction();
	test_multiple_origin_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-statemux", test_main)
