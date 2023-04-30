/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/comparison.hpp>

namespace jive {

#define DEFINE_BITCOMPARISON_OPERATION(NAME, FLAGS, DEBUG_STRING) \
bit ## NAME ## _op::~bit ## NAME ## _op() noexcept \
{} \
\
bool \
bit ## NAME ## _op::operator==(const operation & other) const noexcept \
{ \
	auto op = dynamic_cast<const bit ## NAME ## _op *>(&other); \
	return op && op->type() == type(); \
} \
\
compare_result \
bit ## NAME ## _op::reduce_constants( \
	const bitvalue_repr & arg1, \
	const bitvalue_repr & arg2) const \
{ \
	switch (arg1.NAME(arg2)) { \
		case '0': return compare_result::static_false; \
		case '1': return compare_result::static_true; \
		default: return compare_result::undecidable; \
	} \
} \
\
enum binary_op::flags \
bit ## NAME ## _op::flags() const noexcept \
{ \
	return FLAGS; \
} \
\
std::string \
bit ## NAME ## _op::debug_string() const \
{ \
	return jive::detail::strfmt(#DEBUG_STRING, type().nbits()); \
} \
\
std::unique_ptr<jive::operation> \
bit ## NAME ## _op::copy() const \
{ \
	return std::unique_ptr<jive::operation>(new bit ## NAME ## _op(*this)); \
} \
\
std::unique_ptr<bitcompare_op> \
bit ## NAME ## _op::create(size_t nbits) const \
{ \
	return std::unique_ptr<bitcompare_op>(new bit ## NAME ## _op(nbits)); \
} \

DEFINE_BITCOMPARISON_OPERATION(eq, binary_op::flags::commutative, BITEQ)
DEFINE_BITCOMPARISON_OPERATION(ne, binary_op::flags::commutative, BITNE)
DEFINE_BITCOMPARISON_OPERATION(sge, binary_op::flags::none, BITSGE)
DEFINE_BITCOMPARISON_OPERATION(sgt, binary_op::flags::none, BITSGT)
DEFINE_BITCOMPARISON_OPERATION(sle, binary_op::flags::none, BITSLE)
DEFINE_BITCOMPARISON_OPERATION(slt, binary_op::flags::none, BITSLT)
DEFINE_BITCOMPARISON_OPERATION(uge, binary_op::flags::none, BITUGE)
DEFINE_BITCOMPARISON_OPERATION(ugt, binary_op::flags::none, BITUGT)
DEFINE_BITCOMPARISON_OPERATION(ule, binary_op::flags::none, BITULE)
DEFINE_BITCOMPARISON_OPERATION(ult, binary_op::flags::none, BITULT)

}
