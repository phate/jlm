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
	bitliteral xl(jive::bits::value_repr(4, 1));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_sext", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 1);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr().nbits() == 8);


	xl = jive::bits::value_repr(4, 0x8);
	result = std::move(eval(graph, "test_sext", {&xl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 0xF8);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr().nbits() == 8);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-sext", verify);
