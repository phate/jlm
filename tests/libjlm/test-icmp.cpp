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

#define MAKE_INTOP_VERIFIER(NAME, OP) \
static void \
verify_int##NAME##_op(const jive_graph * graph, uint64_t x, uint64_t y, uint64_t z) \
{ \
	using namespace jive::evaluator; \
\
	memliteral state; \
	bitliteral xl(jive::bits::value_repr(64, x)); \
	bitliteral yl(jive::bits::value_repr(64, y)); \
\
	std::unique_ptr<const literal> result; \
	result = std::move(eval(graph, "test_int" #NAME, {&xl, &yl, &state})->copy()); \
\
	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get()); \
  assert(fctlit->nresults() == 2); \
  assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == z); \
} \

MAKE_INTOP_VERIFIER(slt, slt_op);
MAKE_INTOP_VERIFIER(ult, ult_op);
MAKE_INTOP_VERIFIER(sle, sle_op);
MAKE_INTOP_VERIFIER(ule, ule_op);
MAKE_INTOP_VERIFIER(eq, eq_op);
MAKE_INTOP_VERIFIER(ne, ne_op);
MAKE_INTOP_VERIFIER(sgt, sgt_op);
MAKE_INTOP_VERIFIER(ugt, ugt_op);
MAKE_INTOP_VERIFIER(sge, sge_op);
MAKE_INTOP_VERIFIER(uge, uge_op);

#define MAKE_PTROP_VERIFIER(NAME, OP) \
static void \
verify_ptr##NAME##_op(const jive_graph * graph, uint64_t x, uint64_t y, uint64_t z) \
{ \
	/* FIXME: add checks */ \
} \

MAKE_PTROP_VERIFIER(slt, slt_op);
MAKE_PTROP_VERIFIER(ult, ult_op);
MAKE_PTROP_VERIFIER(sle, sle_op);
MAKE_PTROP_VERIFIER(ule, ule_op);
MAKE_PTROP_VERIFIER(eq, eq_op);
MAKE_PTROP_VERIFIER(ne, ne_op);
MAKE_PTROP_VERIFIER(sgt, sgt_op);
MAKE_PTROP_VERIFIER(ugt, ugt_op);
MAKE_PTROP_VERIFIER(sge, sge_op);
MAKE_PTROP_VERIFIER(uge, uge_op);

static int
verify(const jive_graph * graph)
{
	verify_intslt_op(graph, -3, 4, 1);
	verify_intult_op(graph, 3, 4, 1);
	verify_intsle_op(graph, -3, -3, 1);
	verify_intule_op(graph, -2, -3, 0);
	verify_inteq_op(graph, 4, 5, 0);
	verify_intne_op(graph, 4, 5, 1);
	verify_intsgt_op(graph, -4, -5, 1);
	verify_intugt_op(graph, 4, 5, 0);
	verify_intsge_op(graph, -4, -4, 1);
	verify_intuge_op(graph, 4, 4, 1);

	verify_ptrslt_op(graph, -3, 4, 1);
	verify_ptrult_op(graph, 3, 4, 1);
	verify_ptrsle_op(graph, -3, -3, 1);
	verify_ptrule_op(graph, -2, -3, 0);
	verify_ptreq_op(graph, 4, 5, 0);
	verify_ptrne_op(graph, 4, 5, 1);
	verify_ptrsgt_op(graph, -4, -5, 1);
	verify_ptrugt_op(graph, 4, 5, 0);
	verify_ptrsge_op(graph, -4, -4, 1);
	verify_ptruge_op(graph, 4, 4, 1);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-icmp", nullptr, verify);
