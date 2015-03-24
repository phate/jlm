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

	/* fib(0) */
	memliteral state;
	bitliteral xl(jive::bits::value_repr(32, 0));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "fib", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 0);

	/* fib(1) */
	xl = jive::bits::value_repr(32, 1);
	result = std::move(eval(graph, "fib", {&xl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 1);

	/* fib(2) */
	xl = jive::bits::value_repr(32, 2);
	result = std::move(eval(graph, "fib", {&xl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 1);

	/* fib(3) */
	xl = jive::bits::value_repr(32, 3);
	result = std::move(eval(graph, "fib", {&xl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 2);

	/* fib(8) */
	xl = jive::bits::value_repr(32, 8);
	result = std::move(eval(graph, "fib", {&xl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 21);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-fibrec", verify);
