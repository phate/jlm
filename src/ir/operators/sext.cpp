/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/sext.hpp>

namespace jlm {

sext_op::~sext_op()
{}

bool
sext_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const sext_op*>(&other);
	return op && op->oport_ == oport_ && op->rport_ == rport_;
}

size_t
sext_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
sext_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return oport_;
}

size_t
sext_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
sext_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return rport_;
}

std::string
sext_op::debug_string() const
{
	return strfmt("SEXT[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<jive::operation>
sext_op::copy() const
{
	return std::unique_ptr<jive::operation>(new sext_op(*this));
}

jive_unop_reduction_path_t
sext_op::can_reduce_operand(const jive::output * operand) const noexcept
{
	if (jive::is_bitconstant_node(producer(operand)))
		return jive_unop_reduction_constant;

	return jive_unop_reduction_none;
}

jive::output *
sext_op::reduce_operand(
	jive_unop_reduction_path_t path,
	jive::output * operand) const
{
	if (path == jive_unop_reduction_constant) {
		auto c = static_cast<const jive::bits::constant_op*>(&producer(operand)->operation());
		return create_bitconstant(operand->node()->region(), c->value().sext(ndstbits()-nsrcbits()));
	}

	return nullptr;
}

}
