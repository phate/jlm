/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jive/types/bitstring/arithmetic.h>
#include <jive/view.h>

#include <jlm/ir/operators/sext.hpp>

static inline void
test_bitunary_reduction()
{
	jive::bits::type bt32(32);

	jive::graph graph;
	auto nf = jlm::sext_op::normal_form(&graph);
	nf->set_mutable(false);

	auto x = graph.import(bt32, "x");

	auto y = jive::bits::create_not(32, x);
	auto z = jlm::create_sext(64, y);

	auto ex = graph.export_port(z, "x");

	jive::view(graph, stdout);

	nf->set_mutable(true);
	graph.normalize();
	graph.prune();

	jive::view(graph, stdout);

	assert(jive::bits::is_not_node(ex->origin()->node()));
}

static int
test()
{
	test_bitunary_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-sext", test);
