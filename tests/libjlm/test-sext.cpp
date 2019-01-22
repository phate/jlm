/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jive/types/bitstring/arithmetic.h>
#include <jive/view.h>

#include <jlm/jlm/ir/operators/operators.hpp>
#include <jlm/jlm/ir/operators/sext.hpp>

static inline void
test_bitunary_reduction()
{
	jive::bittype bt32(32);

	jive::graph graph;
	auto nf = jlm::sext_op::normal_form(&graph);
	nf->set_mutable(false);

	auto x = graph.add_import(bt32, "x");

	auto y = jive::bitnot_op::create(32, x);
	auto z = jlm::create_sext(64, y);

	auto ex = graph.add_export(z, "x");

	//jive::view(graph, stdout);

	nf->set_mutable(true);
	graph.normalize();
	graph.prune();

	//jive::view(graph, stdout);

	assert(jive::is<jive::bitnot_op>(ex->origin()->node()));
}

static inline void
test_bitbinary_reduction()
{
	jive::bittype bt32(32);

	jive::graph graph;
	auto nf = jlm::sext_op::normal_form(&graph);
	nf->set_mutable(false);

	auto x = graph.add_import(bt32, "x");
	auto y = graph.add_import(bt32, "y");

	auto z = jive::bitadd_op::create(32, x, y);
	auto w = jlm::create_sext(64, z);

	auto ex = graph.add_export(w, "x");

//	jive::view(graph, stdout);

	nf->set_mutable(true);
	graph.normalize();
	graph.prune();

//	jive::view(graph, stdout);

	assert(jive::is<jive::bitadd_op>(ex->origin()->node()));
}

static inline void
test_inverse_reduction()
{
	jive::bittype bt64(64);

	jive::graph graph;
	auto nf = jlm::sext_op::normal_form(&graph);
	nf->set_mutable(false);

	auto x = graph.add_import(bt64, "x");

	auto y = jlm::create_trunc(32, x);
	auto z = jlm::create_sext(64, y);

	auto ex = graph.add_export(z, "x");

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

JLM_UNIT_TEST_REGISTER("libjlm/test-sext", test)
