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
	std::unique_ptr<const literal> result;

	/* fac(0) */
	bitliteral xl(jive::bits::value_repr(32, 0));
	result = std::move(eval(graph, "fac", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 1);

	/* fac(1) */
	bitliteral yl(jive::bits::value_repr(32, 1));
	result = std::move(eval(graph, "fac", {&yl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 1);

	/* fac(5) */
	bitliteral zl(jive::bits::value_repr(32, 5));
	result = std::move(eval(graph, "fac", {&zl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 120);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-while", nullptr, verify);
