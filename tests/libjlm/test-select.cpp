/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>

#include <assert.h>

static int
verify(const jive_graph * graph)
{
	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(32, 13));
	bitliteral yl(jive::bits::value_repr(32, 14));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "max", {&xl, &yl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 14);

	/* exchange arguments */
	result = std::move(eval(graph, "max", {&yl, &xl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 14);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-select", verify);
