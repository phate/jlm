/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>
#include <jive/types/bitstring/arithmetic.h>
#include <jive/types/bitstring/comparison.h>
#include <jive/types/bitstring/type.h>

#include <assert.h>

#define MAKE_OP_VERIFIER(NAME, OP) \
static void \
verify_##NAME##_op(const jive_graph * graph, uint64_t x, uint64_t y, uint64_t z) \
{ \
	using namespace jive::evaluator; \
\
	memliteral state; \
	bitliteral xl(jive::bits::value_repr(64, x)); \
	bitliteral yl(jive::bits::value_repr(64, y)); \
\
	std::unique_ptr<const literal> result; \
	result = std::move(eval(graph, "test_" #NAME, {&xl, &yl, &state})->copy()); \
\
	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get()); \
  assert(fctlit->nresults() == 2); \
  assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == z); \
} \

MAKE_OP_VERIFIER(slt, slt_op);
MAKE_OP_VERIFIER(ult, ult_op);
MAKE_OP_VERIFIER(sle, sle_op);
MAKE_OP_VERIFIER(ule, ule_op);
MAKE_OP_VERIFIER(eq, eq_op);
MAKE_OP_VERIFIER(ne, ne_op);
MAKE_OP_VERIFIER(sgt, sgt_op);
MAKE_OP_VERIFIER(ugt, ugt_op);
MAKE_OP_VERIFIER(sge, sge_op);
MAKE_OP_VERIFIER(uge, uge_op);

static int
verify(const jive_graph * graph)
{
	verify_slt_op(graph, -3, 4, 1);
	verify_ult_op(graph, 3, 4, 1);
	verify_sle_op(graph, -3, -3, 1);
	verify_ule_op(graph, -2, -3, 0);
	verify_eq_op(graph, 4, 5, 0);
	verify_ne_op(graph, 4, 5, 1);
	verify_sgt_op(graph, -4, -5, 1);
	verify_ugt_op(graph, 4, 5, 0);
	verify_sge_op(graph, -4, -4, 1);
	verify_uge_op(graph, 4, 4, 1);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-icmp", verify);
