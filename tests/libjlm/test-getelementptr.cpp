/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
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
verify(const jive_graph * graph)
{
	/* FIXME: remove when the evaluator understands the address type */
	setlocale(LC_ALL, "");

	jive_memlayout_mapper_simple mapper;
	jive_memlayout_mapper_simple_init(&mapper, 64);
	jive_graph_address_transform(const_cast<jive_graph*>(graph), &mapper.base.base);
	jive_memlayout_mapper_simple_fini(&mapper);

	jive_graph_normalize(const_cast<jive_graph*>(graph));
	jive_graph_prune(const_cast<jive_graph*>(graph));
	jive_view(const_cast<jive_graph*>(graph), stdout);

	using namespace jive::evaluator;

	uint64_t array[] = {3, 4};
	memliteral state;
	bitliteral xl(jive::bits::value_repr(64, (uint64_t)array));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_getelementptr", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 7);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-getelementptr", nullptr, verify);
