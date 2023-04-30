/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

namespace jive {

/* bitunary operation */

bitunary_op::~bitunary_op() noexcept
{}

jive_binop_reduction_path_t
bitunary_op::can_reduce_operand(
	const jive::output * arg) const noexcept
{
	if (is<bitconstant_op>(producer(arg)))
		return jive_unop_reduction_constant;

	return jive_unop_reduction_none;
}

jive::output *
bitunary_op::reduce_operand(
	jive_unop_reduction_path_t path,
	jive::output * arg) const
{
	if (path == jive_unop_reduction_constant) {
		auto p = producer(arg);
		auto & c = static_cast<const bitconstant_op&>(p->operation());
		return create_bitconstant(p->region(), reduce_constant(c.value()));
	}

	return nullptr;
}

/* bitbinary operation */

bitbinary_op::~bitbinary_op() noexcept
{}

jive_binop_reduction_path_t
bitbinary_op::can_reduce_operand_pair(
	const jive::output * arg1,
	const jive::output * arg2) const noexcept
{
	if (is<bitconstant_op>(producer(arg1)) && is<bitconstant_op>(producer(arg2)))
		return jive_binop_reduction_constants;

	return jive_binop_reduction_none;
}

jive::output *
bitbinary_op::reduce_operand_pair(
	jive_binop_reduction_path_t path,
	jive::output * arg1,
	jive::output * arg2) const
{
	if (path == jive_binop_reduction_constants) {
		auto & c1 = static_cast<const bitconstant_op&>(producer(arg1)->operation());
		auto & c2 = static_cast<const bitconstant_op&>(producer(arg2)->operation());
		return create_bitconstant(arg1->region(), reduce_constants(c1.value(), c2.value()));
	}

	return nullptr;
}

/* bitcompare operation */

bitcompare_op::~bitcompare_op() noexcept
{}

jive_binop_reduction_path_t
bitcompare_op::can_reduce_operand_pair(
	const jive::output * arg1,
	const jive::output * arg2) const noexcept
{
	auto p = producer(arg1);
	const bitconstant_op * c1_op = nullptr;
	if (p) c1_op = dynamic_cast<const bitconstant_op*>(&p->operation());

	p = producer(arg2);
	const bitconstant_op * c2_op = nullptr;
	if (p) c2_op = dynamic_cast<const bitconstant_op*>(&p->operation());

	bitvalue_repr arg1_repr = c1_op ? c1_op->value() : bitvalue_repr::repeat(type().nbits(), 'D');
	bitvalue_repr arg2_repr = c2_op ? c2_op->value() : bitvalue_repr::repeat(type().nbits(), 'D');

	switch (reduce_constants(arg1_repr, arg2_repr)) {
		case compare_result::static_false:
			return 1;
		case compare_result::static_true:
			return 2;
		case compare_result::undecidable:
			return jive_binop_reduction_none;
	}
	
	return jive_binop_reduction_none;
}

jive::output *
bitcompare_op::reduce_operand_pair(
	jive_binop_reduction_path_t path,
	jive::output * arg1,
	jive::output * arg2) const
{
	if (path == 1) {
		return create_bitconstant(arg1->region(), "0");
	}
	if (path == 2) {
		return create_bitconstant(arg1->region(), "1");
	}

	return nullptr;
}

}
