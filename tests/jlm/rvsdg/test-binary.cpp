/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

static void
test_flattened_binary_reduction()
{
	using namespace jive;

	jlm::valuetype vt;
	jlm::binary_op op(vt, vt, binary_op::flags::associative);

	/* test paralell reduction */
	{
		jive::graph graph;
		auto i0 = graph.add_import({vt, ""});
		auto i1 = graph.add_import({vt, ""});
		auto i2 = graph.add_import({vt, ""});
		auto i3 = graph.add_import({vt, ""});

		auto o1 = simple_node::create_normalized(graph.root(), op, {i0, i1})[0];
		auto o2 = simple_node::create_normalized(graph.root(), op, {o1, i2})[0];
		auto o3 = simple_node::create_normalized(graph.root(), op, {o2, i3})[0];

		auto ex = graph.add_export(o3, {o3->type(), ""});
		graph.prune();

		jive::view(graph, stdout);
		assert(graph.root()->nnodes() == 1 && region::Contains<flattened_binary_op>(*graph.root(), false));

		flattened_binary_op::reduce(&graph, jive::flattened_binary_op::reduction::parallel);
		jive::view(graph, stdout);

		assert(graph.root()->nnodes() == 3);

		auto node0 = node_output::node(ex->origin());
		assert(is<jlm::binary_op>(node0));

		auto node1 = node_output::node(node0->input(0)->origin());
		assert(is<jlm::binary_op>(node1));

		auto node2 = node_output::node(node0->input(1)->origin());
		assert(is<jlm::binary_op>(node2));
	}

	/* test linear reduction */
	{
		jive::graph graph;
		auto i0 = graph.add_import({vt, ""});
		auto i1 = graph.add_import({vt, ""});
		auto i2 = graph.add_import({vt, ""});
		auto i3 = graph.add_import({vt, ""});

		auto o1 = simple_node::create_normalized(graph.root(), op, {i0, i1})[0];
		auto o2 = simple_node::create_normalized(graph.root(), op, {o1, i2})[0];
		auto o3 = simple_node::create_normalized(graph.root(), op, {o2, i3})[0];

		auto ex = graph.add_export(o3, {o3->type(), ""});
		graph.prune();

		jive::view(graph, stdout);
		assert(graph.root()->nnodes() == 1 && region::Contains<flattened_binary_op>(*graph.root(), false));

		flattened_binary_op::reduce(&graph, jive::flattened_binary_op::reduction::linear);
		jive::view(graph, stdout);

		assert(graph.root()->nnodes() == 3);

		auto node0 = node_output::node(ex->origin());
		assert(is<jlm::binary_op>(node0));

		auto node1 = node_output::node(node0->input(0)->origin());
		assert(is<jlm::binary_op>(node1));

		auto node2 = node_output::node(node1->input(0)->origin());
		assert(is<jlm::binary_op>(node2));
	}
}

static int
test_main()
{
	test_flattened_binary_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-binary", test_main)
