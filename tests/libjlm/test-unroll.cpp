/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>
#include <jive/types/bitstring/arithmetic.h>
#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/comparison.h>
#include <jive/types/function/fctlambda.h>
#include <jive/view.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/simple_node.h>
#include <jive/vsdg/theta.h>

#include <jlm/opt/dne.hpp>
#include <jlm/opt/unroll.hpp>

static inline void
test1()
{
	jive::bits::type bt(32);
	jlm::test_op op({&bt}, {&bt});

	jive::graph graph;
	auto x = graph.import(bt, "x");
	auto y = graph.import(bt, "y");

	jive::theta_builder tb;
	tb.begin_theta(graph.root());
	auto lv1 = tb.add_loopvar(x);
	auto lv2 = tb.add_loopvar(y);

	auto one = jive::create_bitconstant(tb.subregion(), 32, 1);
	auto add = jive::bits::create_add(32, lv1->argument(), one);
	auto cmp = jive::bits::create_ult(32, add, lv2->argument());
	auto match = jive::ctl::match(1, {{1, 0}}, 1, 2, cmp);

	lv1->result()->divert_origin(add);

	tb.end_theta(match);

	auto ex1 = graph.export_port(lv1->output(), "x");

	jive::view(graph, stdout);
	jlm::unroll(graph, 2);
	jive::view(graph, stdout);

	auto node = ex1->origin()->node();
	assert(jive::is_gamma_node(node));
	node = node->input(1)->origin()->node();
	assert(jive::is_gamma_node(node));
}

static int
verify()
{
	test1();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-unroll", verify);
