/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
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

	bool exception_caught = false;
	try {
		std::unique_ptr<const literal> result;
		result = std::move(eval(graph, "unreachable", {&xl, &state})->copy());
	} catch (jive::compiler_error e) {
		exception_caught = true;
	}

	assert(exception_caught);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-unreachable", nullptr, verify);
