/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/arch/address-transform.h>
#include <jive/arch/memlayout-simple.h>
#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>

#include <jive/view.h>

#include <assert.h>

static int
verify(const jive::graph * graph)
{
#if 0
	/* FIXME: remove when the evaluator understands the address type */
	setlocale(LC_ALL, "");

	jive::memlayout_mapper_simple mapper(8);
	jive_graph_address_transform(const_cast<jive_graph*>(graph), &mapper);

	jive_graph_normalize(const_cast<jive_graph*>(graph));
	jive_graph_prune(const_cast<jive_graph*>(graph));
	jive_view(graph, stdout);

	using namespace jive::evaluator;

	uint64_t x = 3;
	memliteral state;
	bitliteral xl(jive::bits::value_repr(64, (uint64_t)&x));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_loadstore", {&xl, &state})->copy());

	assert(x == 5);
#endif
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-loadstore", nullptr, verify);
