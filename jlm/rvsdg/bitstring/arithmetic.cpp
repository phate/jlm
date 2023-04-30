/*
 * Copyright 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

namespace jive {

#define DEFINE_BITUNARY_OPERATION(NAME, REDUCTION, DEBUG_STRING) \
NAME ## _op::~NAME ## _op() noexcept \
{} \
\
bool \
NAME ## _op::operator==(const operation & other) const noexcept \
{ \
	auto op = dynamic_cast<const NAME ## _op*>(&other); \
	return op && op->type() == type(); \
} \
\
bitvalue_repr \
NAME ## _op::reduce_constant(const bitvalue_repr & arg) const \
{ \
	return REDUCTION; \
} \
\
std::string \
NAME ## _op::debug_string() const \
{ \
	return jive::detail::strfmt(#DEBUG_STRING, type().nbits()); \
} \
\
std::unique_ptr<jive::operation> \
NAME ## _op::copy() const \
{ \
	return std::unique_ptr<jive::operation>(new NAME ## _op(*this)); \
} \
\
std::unique_ptr<bitunary_op> \
NAME ## _op::create(size_t nbits) const \
{ \
	return std::unique_ptr<bitunary_op>(new NAME ## _op(nbits)); \
} \

#define DEFINE_BITBINARY_OPERATION(NAME, REDUCTION, DEBUG_STRING, FLAGS) \
NAME ## _op::~NAME ## _op() noexcept \
{} \
\
bool \
NAME ## _op::operator==(const operation & other) const noexcept \
{ \
	auto op = dynamic_cast<const NAME ## _op*>(&other); \
	return op && op->type() == type(); \
} \
\
bitvalue_repr \
NAME ## _op::reduce_constants( \
	const bitvalue_repr & arg1, \
	const bitvalue_repr & arg2) const \
{ \
	return REDUCTION; \
} \
\
enum binary_op::flags \
NAME ## _op::flags() const noexcept \
{ \
	return FLAGS; \
} \
\
std::string \
NAME ## _op::debug_string() const \
{ \
	return jive::detail::strfmt(#DEBUG_STRING, type().nbits()); \
} \
\
std::unique_ptr<jive::operation> \
NAME ## _op::copy() const \
{ \
	return std::unique_ptr<jive::operation>(new NAME ## _op(*this)); \
} \
\
std::unique_ptr<bitbinary_op> \
NAME ## _op::create(size_t nbits) const \
{ \
	return std::unique_ptr<bitbinary_op>(new NAME ## _op(nbits)); \
} \

DEFINE_BITUNARY_OPERATION(bitneg, arg.neg(), BITNEGATE)
DEFINE_BITUNARY_OPERATION(bitnot, arg.lnot(), BITNOT)

DEFINE_BITBINARY_OPERATION(bitadd, arg1.add(arg2), BITADD,
	binary_op::flags::associative | binary_op::flags::commutative)
DEFINE_BITBINARY_OPERATION(bitand, arg1.land(arg2), BITAND,
	binary_op::flags::associative | binary_op::flags::commutative)
DEFINE_BITBINARY_OPERATION(bitashr, arg1.ashr(arg2.to_uint()), BITASHR, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitmul, arg1.mul(arg2), BITMUL,
	binary_op::flags::associative | binary_op::flags::commutative)
DEFINE_BITBINARY_OPERATION(bitor, arg1.lor(arg2), BITOR,
	binary_op::flags::associative | binary_op::flags::commutative)
DEFINE_BITBINARY_OPERATION(bitsdiv, arg1.sdiv(arg2), BITSDIV, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitshl, arg1.shl(arg2.to_uint()), BITSHL, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitshr, arg1.shr(arg2.to_uint()), BITSHR, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitsmod, arg1.smod(arg2), BITSMOD, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitsmulh, arg1.smulh(arg2), BITSMULH, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitsub, arg1.sub(arg2), BITSUB, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitudiv, arg1.udiv(arg2), BITUDIV, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitumod, arg1.umod(arg2), BITUMOD, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitumulh, arg1.umulh(arg2), BITUMULH, binary_op::flags::none)
DEFINE_BITBINARY_OPERATION(bitxor, arg1.lxor(arg2), BITXOR,
	binary_op::flags::associative | binary_op::flags::commutative)

}
