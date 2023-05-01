/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

static void
test_node_copy(void)
{
	using namespace jive;

	jlm::statetype stype;
	jlm::valuetype vtype;

	jive::graph graph;
	auto s = graph.add_import({stype, ""});
	auto v = graph.add_import({vtype, ""});

	auto n1 = jlm::structural_node::create(graph.root(), 3);
	auto i1 = structural_input::create(n1, s, stype);
	auto i2 = structural_input::create(n1, v, vtype);
	auto o1 = structural_output::create(n1, stype);
	auto o2 = structural_output::create(n1, vtype);

	auto a1 = argument::create(n1->subregion(0), i1, stype);
	auto a2 = argument::create(n1->subregion(0), i2, vtype);

	auto n2 = jlm::test_op::create(n1->subregion(0), {a1}, {&stype});
	auto n3 = jlm::test_op::create(n1->subregion(0), {a2}, {&vtype});

	result::create(n1->subregion(0), n2->output(0), o1, stype);
	result::create(n1->subregion(0), n3->output(0), o2, vtype);

	jive::view(graph.root(), stdout);

	/* copy first into second region with arguments and results */
	substitution_map smap;
	smap.insert(i1, i1); smap.insert(i2, i2);
	smap.insert(o1, o1); smap.insert(o2, o2);
	n1->subregion(0)->copy(n1->subregion(1), smap, true, true);

	jive::view(graph.root(), stdout);

	auto r2 = n1->subregion(1);
	assert(r2->narguments() == 2);
	assert(r2->argument(0)->input() == i1);
	assert(r2->argument(1)->input() == i2);

	assert(r2->nresults() == 2);
	assert(r2->result(0)->output() == o1);
	assert(r2->result(1)->output() == o2);

	assert(r2->nnodes() == 2);

	/* copy second into third region only with arguments */
	jive::substitution_map smap2;
	auto a3 = argument::create(n1->subregion(2), i1, stype);
	auto a4 = argument::create(n1->subregion(2), i2, vtype);
	smap2.insert(r2->argument(0), a3);
	smap2.insert(r2->argument(1), a4);

	smap2.insert(o1, o1); smap2.insert(o2, o2);
	n1->subregion(1)->copy(n1->subregion(2), smap2, false, true);

	jive::view(graph.root(), stdout);

	auto r3 = n1->subregion(2);
	assert(r3->nresults() == 2);
	assert(r3->result(0)->output() == o1);
	assert(r3->result(1)->output() == o2);

	assert(r3->nnodes() == 2);

	/* copy structural node */
	jive::substitution_map smap3;
	smap3.insert(s, s); smap3.insert(v, v);
	n1->copy(graph.root(), smap3);

	jive::view(graph.root(), stdout);

	assert(graph.root()->nnodes() == 2);
}

static inline void
test_node_depth()
{
	jlm::valuetype vt;

	jive::graph graph;
	auto x = graph.add_import({vt, "x"});

	auto null = jlm::test_op::create(graph.root(), {}, {&vt});
	auto bin = jlm::test_op::create(graph.root(), {null->output(0), x}, {&vt});
	auto un = jlm::test_op::create(graph.root(), {bin->output(0)}, {&vt});

	graph.add_export(un->output(0), {un->output(0)->type(), "x"});

	jive::view(graph.root(), stdout);

	assert(null->depth() == 0);
	assert(bin->depth() == 1);
	assert(un->depth() == 2);

	bin->input(0)->divert_to(x);
	assert(bin->depth() == 0);
	assert(un->depth() == 1);
}

static int
test_nodes()
{
	test_node_copy();
	test_node_depth();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-nodes", test_nodes)
