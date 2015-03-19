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

	/* case 7 */
	memliteral state;
	bitliteral xl(jive::bits::value_repr(32, 7));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_switch0", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 5);


	/* case 8 */
	bitliteral yl(jive::bits::value_repr(32, 8));
	result = std::move(eval(graph, "test_switch0", {&yl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 6);


	/* default case */
	bitliteral zl(jive::bits::value_repr(32, 0));
	result = std::move(eval(graph, "test_switch0", {&zl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 11);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-switch0", verify);
