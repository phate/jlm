/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>

static inline void
test_bitunary_reduction()
{
	jive::bittype bt32(32);

	jive::graph graph;
	auto nf = jlm::sext_op::normal_form(&graph);
	nf->set_mutable(false);

	auto x = graph.add_import({bt32, "x"});

	auto y = jive::bitnot_op::create(32, x);
	auto z = jlm::sext_op::create(64, y);

	auto ex = graph.add_export(z, {z->type(), "x"});

	//jive::view(graph, stdout);

	nf->set_mutable(true);
	graph.normalize();
	graph.prune();

	//jive::view(graph, stdout);

	assert(jive::is<jive::bitnot_op>(jive::node_output::node(ex->origin())));
}

static inline void
test_bitbinary_reduction()
{
	jive::bittype bt32(32);

	jive::graph graph;
	auto nf = jlm::sext_op::normal_form(&graph);
	nf->set_mutable(false);

	auto x = graph.add_import({bt32, "x"});
	auto y = graph.add_import({bt32, "y"});

	auto z = jive::bitadd_op::create(32, x, y);
	auto w = jlm::sext_op::create(64, z);

	auto ex = graph.add_export(w, {w->type(), "x"});

//	jive::view(graph, stdout);

	nf->set_mutable(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph, stdout);

	assert(jive::is<jive::bitadd_op>(jive::node_output::node(ex->origin())));
}

static inline void
test_inverse_reduction()
{
	using namespace jlm;

	jive::bittype bt64(64);

	jive::graph graph;
	auto nf = jlm::sext_op::normal_form(&graph);
	nf->set_mutable(false);

	auto x = graph.add_import({bt64, "x"});

	auto y = trunc_op::create(32, x);
	auto z = jlm::sext_op::create(64, y);

	auto ex = graph.add_export(z, {z->type(), "x"});

	jive::view(graph, stdout);

	nf->set_mutable(true);
	graph.normalize();
	graph.prune();

	jive::view(graph, stdout);

	assert(ex->origin() == x);
}

static int
test()
{
	test_bitunary_reduction();
	test_bitbinary_reduction();
	test_inverse_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/test-sext", test)
