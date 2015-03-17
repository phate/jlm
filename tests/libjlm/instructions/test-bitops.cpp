/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/types/bitstring/arithmetic.h>
#include <jive/types/bitstring/comparison.h>
#include <jive/types/bitstring/type.h>

#include <assert.h>

#define MAKE_OP_VERIFIER(NAME, OP) \
static void \
verify_##NAME##_op(const jive_graph * graph) \
{ \
	/*FIXME*/ \
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
	verify_add_op(graph);
	verify_and_op(graph);
	verify_ashr_op(graph);
	verify_sub_op(graph);
	verify_udiv_op(graph);
	verify_sdiv_op(graph);
	verify_urem_op(graph);
	verify_srem_op(graph);
	verify_shl_op(graph);
	verify_lshr_op(graph);
	verify_or_op(graph);
	verify_xor_op(graph);
	verify_mul_op(graph);

	verify_slt_op(graph);
	verify_ult_op(graph);
	verify_sle_op(graph);
	verify_ule_op(graph);
	verify_eq_op(graph);
	verify_ne_op(graph);
	verify_sgt_op(graph);
	verify_ugt_op(graph);
	verify_sge_op(graph);
	verify_uge_op(graph);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-bitops", verify);
