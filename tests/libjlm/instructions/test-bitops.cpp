/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/tac/tac.hpp>

#include <jive/types/bitstring/arithmetic.h>
#include <jive/types/bitstring/comparison.h>
#include <jive/types/bitstring/type.h>

#include <assert.h>

/*
	FIXME: test does not check for bitops
*/

#define MAKE_OP_VERIFIER(NAME, OP) \
static void \
verify_##NAME##_op(jlm::frontend::clg & clg) \
{ \
	jlm::frontend::clg_node * node = clg.lookup_function("test_" #NAME); \
	assert(node != nullptr); \
\
	jlm::frontend::cfg * cfg = node->cfg(); \
\
	assert(cfg->is_linear()); \
\
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
verify(jlm::frontend::clg & clg)
{
	verify_add_op(clg);
	verify_and_op(clg);
	verify_ashr_op(clg);
	verify_sub_op(clg);
	verify_udiv_op(clg);
	verify_sdiv_op(clg);
	verify_urem_op(clg);
	verify_srem_op(clg);
	verify_shl_op(clg);
	verify_lshr_op(clg);
	verify_or_op(clg);
	verify_xor_op(clg);
	verify_mul_op(clg);

	verify_slt_op(clg);
	verify_ult_op(clg);
	verify_sle_op(clg);
	verify_ule_op(clg);
	verify_eq_op(clg);
	verify_ne_op(clg);
	verify_sgt_op(clg);
	verify_ugt_op(clg);
	verify_sge_op(clg);
	verify_uge_op(clg);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-bitops", verify);
