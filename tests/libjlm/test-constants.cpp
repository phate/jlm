/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>

#include <jive/view.h>

#include <assert.h>

static int
verify_constantInt(const jive_graph * graph)
{
	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(32, 13));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_constantInt", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 12);

	return 0;
}

static int
verify_constantFP(const jive_graph * graph)
{
	jive_view(const_cast<jive_graph*>(graph), stdout);

	/* FIXME: insert checks for all types */

	return 0;
}

static int
verify_constantPointerNull(const jive_graph * graph)
{
	/* FIXME: insert checks */

	return 0;
}

static int
verify_globalVariable(const jive_graph * graph)
{
	/* FIXME: insert checks */

	return 0;
}

static int
verify_undefValue(const jive_graph * graph)
{
	using namespace jive::evaluator;

	memliteral state;
	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_undefValue", {&state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(
		&fctlit->result(0))->value_repr() == jive::bits::value_repr::repeat(32, 'X'));

	return 0;
}

static int
verify_constantAggregateZeroStruct(const jive_graph * graph)
{
	/* FIXME: insert checks */

	return 0;
}

static int
verify(const jive_graph * graph)
{
	verify_constantFP(graph);
	verify_constantInt(graph);
	verify_constantPointerNull(graph);
	verify_globalVariable(graph);
	verify_undefValue(graph);
	verify_constantAggregateZeroStruct(graph);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-constants", verify);
