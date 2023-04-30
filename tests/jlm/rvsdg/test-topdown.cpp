/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/traverser.hpp>

static void
test_initialization()
{
	jlm::valuetype vtype;

	jive::graph graph;
	auto i = graph.add_import({vtype, "i"});

	auto constant = jlm::test_op::create(graph.root(), {}, {&vtype});
	auto unary = jlm::test_op::create(graph.root(), {i}, {&vtype});
	auto binary = jlm::test_op::create(graph.root(), {i, unary->output(0)}, {&vtype});

	graph.add_export(constant->output(0), {constant->output(0)->type(), "c"});
	graph.add_export(unary->output(0), {unary->output(0)->type(), "u"});
	graph.add_export(binary->output(0), {binary->output(0)->type(), "b"});

	bool unary_visited = false;
	bool binary_visited = false;
	bool constant_visited = false;
	for (const auto & node : jive::topdown_traverser(graph.root())) {
		if (node == unary) unary_visited = true;
		if (node == constant) constant_visited = true;
		if (node == binary && unary_visited) binary_visited = true;
	}

	assert(unary_visited);
	assert(binary_visited);
	assert(constant_visited);
}

static void
test_basic_traversal()
{
	jive::graph graph;
	jlm::valuetype type;

	auto n1 = jlm::test_op::create(graph.root(), {}, {&type, &type});
	auto n2 = jlm::test_op::create(graph.root(), {n1->output(0), n1->output(1)}, {&type});

	graph.add_export(n2->output(0), {n2->output(0)->type(), "dummy"});

	{
		jive::node * tmp;
		jive::topdown_traverser trav(graph.root());

		tmp = trav.next();
		assert(tmp == n1);
		tmp = trav.next();
		assert(tmp == n2);
		tmp = trav.next();
		assert(tmp == nullptr);
	}

	assert(!has_active_trackers(&graph));
}

static void
test_order_enforcement_traversal()
{
	jive::graph graph;
	jlm::valuetype type;

	auto n1 = jlm::test_op::create(graph.root(), {}, {&type, &type});
	auto n2 = jlm::test_op::create(graph.root(), {n1->output(0)}, {&type});
	auto n3 = jlm::test_op::create(graph.root(), {n2->output(0),n1->output(1)}, {&type});

	{
		jive::node * tmp;
		jive::topdown_traverser trav(graph.root());

		tmp = trav.next();
		assert(tmp == n1);
		tmp = trav.next();
		assert(tmp == n2);
		tmp = trav.next();
		assert(tmp == n3);
		tmp = trav.next();
		assert(tmp == nullptr);
	}

	assert(!has_active_trackers(&graph));
}

static void
test_traversal_insertion()
{
	jive::graph graph;
	jlm::valuetype type;

	auto n1 = jlm::test_op::create(graph.root(), {}, {&type, &type});
	auto n2 = jlm::test_op::create(graph.root(), {n1->output(0), n1->output(1)}, {&type});

	graph.add_export(n2->output(0), {n2->output(0)->type(), "dummy"});

	{
		jive::node * node;
		jive::topdown_traverser trav(graph.root());

		node = trav.next();
		assert(node == n1);

		/* At this point, n1 has been visited, now create some nodes */

		auto n3 = jlm::test_op::create(graph.root(), {}, {&type});
		auto n4 = jlm::test_op::create(graph.root(), {n3->output(0)}, {});
		auto n5 = jlm::test_op::create(graph.root(), {n2->output(0)}, {});

		/*
			The newly created nodes n3 and n4 will not be visited,
			as they were not created as descendants of unvisited
			nodes. n5 must be visited, as n2 has not been visited yet.
		*/

		bool visited_n2 = false, visited_n3 = false, visited_n4 = false, visited_n5 = false;
		for(;;) {
			node = trav.next();
			if (!node) break;
			if (node == n2) visited_n2 = true;
			if (node == n3) visited_n3 = true;
			if (node == n4) visited_n4 = true;
			if (node == n5) visited_n5 = true;
		}

		assert(visited_n2);
		assert(!visited_n3);
		assert(!visited_n4);
		assert(visited_n5);
	}

	assert(!has_active_trackers(&graph));
}

static void
test_mutable_traverse()
{
	auto test = [](jive::graph * graph, jive::node * n1, jive::node * n2, jive::node * n3) {
		bool seen_n1 = false;
		bool seen_n2 = false;
		bool seen_n3 = false;

		for (const auto & tmp : jive::topdown_traverser(graph->root())) {
			seen_n1 = seen_n1 || (tmp == n1);
			seen_n2 = seen_n2 || (tmp == n2);
			seen_n3 = seen_n3 || (tmp == n3);
			if (n3->input(0)->origin() == n1->output(0))
				n3->input(0)->divert_to(n2->output(0));
			else
				n3->input(0)->divert_to(n1->output(0));
		}

		assert(seen_n1);
		assert(seen_n2);
		assert(seen_n3);
	};

	jive::graph graph;
	jlm::valuetype type;
	auto n1 = jlm::test_op::create(graph.root(), {}, {&type});
	auto n2 = jlm::test_op::create(graph.root(), {}, {&type});
	auto n3 = jlm::test_op::create(graph.root(), {n1->output(0)}, {});

	test(&graph, n1, n2, n3);
	test(&graph, n1, n2, n3);
}

static int
test_main(void)
{
	test_initialization();
	test_basic_traversal();
	test_order_enforcement_traversal();
	test_traversal_insertion();
	test_mutable_traverse();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-topdown", test_main)
