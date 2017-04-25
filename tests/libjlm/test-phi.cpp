/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>

#include <assert.h>

static int
verify(const jive::graph * graph)
{
	using namespace jive::evaluator;

	memliteral state;

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_phi", {&state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 100);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-phi", nullptr, verify);
