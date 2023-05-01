/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <assert.h>
#include <stdio.h>

#include <jlm/rvsdg/view.hpp>

static bool
region_contains_node(const jive::region * region, const jive::node * n)
{
	for (const auto & node : region->nodes) {
		if (&node == n)
			return true;
	}

	return false;
}

static int
test_recursive_prune()
{
	using namespace jive;

	jlm::valuetype t;

	jive::graph graph;
	auto imp = graph.add_import({t, "i"});

	auto n1 = jlm::test_op::create(graph.root(), {imp}, {&t});
	auto n2 = jlm::test_op::create(graph.root(), {imp}, {&t});

	auto n3 = jlm::structural_node::create(graph.root(), 1);
	structural_input::create(n3, imp, t);
	auto a1 = argument::create(n3->subregion(0), nullptr, t);
	auto n4 = jlm::test_op::create(n3->subregion(0), {a1}, {&t});
	auto n5 = jlm::test_op::create(n3->subregion(0), {a1}, {&t});
	result::create(n3->subregion(0), n4->output(0), nullptr, t);
	auto o1 = structural_output::create(n3, t);

	auto n6 = jlm::structural_node::create(n3->subregion(0), 1);

	graph.add_export(n2->output(0), {n2->output(0)->type(), "n2"});
	graph.add_export(o1, {o1->type(), "n3"});

	jive::view(graph.root(), stdout);
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(!region_contains_node(graph.root(), n1));
	assert(region_contains_node(graph.root(), n2));
	assert(region_contains_node(graph.root(), n3));
	assert(region_contains_node(n3->subregion(0), n4));
	assert(!region_contains_node(n3->subregion(0), n5));
	assert(!region_contains_node(n3->subregion(0), n6));

	return 0;
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-graph_prune", test_recursive_prune)

static int
test_empty_graph_pruning(void)
{
	jive::graph graph;

	jive::view(graph.root(), stdout);

	graph.prune();

	assert(graph.root()->nnodes() == 0);

	jive::view(graph.root(), stdout);

	return 0;
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-empty_graph_pruning", test_empty_graph_pruning)

static int
test_prune_replace(void)
{
	using namespace jive;

	jlm::valuetype type;

	jive::graph graph;
	auto n1 = jlm::test_op::create(graph.root(), {}, {&type});
	auto n2 = jlm::test_op::create(graph.root(), {n1->output(0)}, {&type});
	auto n3 = jlm::test_op::create(graph.root(), {n2->output(0)}, {&type});

	graph.add_export(n2->output(0), {n2->output(0)->type(), "n2"});
	graph.add_export(n3->output(0), {n2->output(0)->type(), "n3"});

	auto n4 = jlm::test_op::create(graph.root(), {n1->output(0)}, {&type});

	n2->output(0)->divert_users(n4->output(0));
	assert(n2->output(0)->nusers() == 0);

	graph.prune();

	assert(!region_contains_node(graph.root(), n2));

	return 0;
}

JLM_UNIT_TEST_REGISTER("rvsdg/test-prune-replace", test_prune_replace)

static int
test_graph(void)
{
	using namespace jive;

	jlm::valuetype type;

	jive::graph graph;
	
	auto n1 = jlm::test_op::create(graph.root(), {}, {&type});
	assert(n1);
	assert(n1->depth() == 0);

	auto n2 = jlm::test_op::create(graph.root(), {n1->output(0)}, {});
	assert(n2);
	assert(n2->depth() == 1);
	
	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-graph", test_graph)
