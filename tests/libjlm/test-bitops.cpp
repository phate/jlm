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
verify_##NAME##_op(const jive::graph * graph, uint64_t x, uint64_t y, uint64_t z) \
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

MAKE_OP_VERIFIER(add, add_op);
MAKE_OP_VERIFIER(and, and_op);
MAKE_OP_VERIFIER(ashr, ashr_op);
MAKE_OP_VERIFIER(sub, sub_op);
MAKE_OP_VERIFIER(udiv, udiv_op);
MAKE_OP_VERIFIER(sdiv, sdiv_op);
MAKE_OP_VERIFIER(urem, umod_op);
MAKE_OP_VERIFIER(srem, smod_op);
MAKE_OP_VERIFIER(shl, shl_op);
MAKE_OP_VERIFIER(lshr, shr_op);
MAKE_OP_VERIFIER(or, or_op);
MAKE_OP_VERIFIER(xor, xor_op);
MAKE_OP_VERIFIER(mul, mul_op);

static int
verify(const jive::graph * graph)
{
	verify_add_op(graph, 3, 4, 7);
	verify_and_op(graph, 3, 6, 2);
	verify_ashr_op(graph, 0x8000000000000001, 1, 0xC000000000000000);
	verify_sub_op(graph, 5, 3, 2);
	verify_udiv_op(graph, 16, 4, 4);
	verify_sdiv_op(graph, -16, 4, -4);
	verify_urem_op(graph, 16, 5, 1);
	verify_srem_op(graph, -16, 5, -1);
	verify_shl_op(graph, 1, 1, 2);
	verify_lshr_op(graph, 2, 1, 1);
	verify_or_op(graph, 3, 6, 7);
	verify_xor_op(graph, 3, 6, 5);
	verify_mul_op(graph, 3, 4, 12);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-bitops", nullptr, verify);
