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

	memliteral state;
	bitliteral xl(jive::bits::value_repr(64, 3));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_ptrtoint", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 3);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-ptrtoint", verify);
