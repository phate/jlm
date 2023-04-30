/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/control.hpp>

namespace jive {

/* control constant */

// explicit instantiation
template class domain_const_op<ctltype, ctlvalue_repr, ctlformat_value, ctltype_of_value>;

/* control type */

ctltype::~ctltype() noexcept
{}

ctltype::ctltype(size_t nalternatives)
: jive::statetype()
, nalternatives_(nalternatives)
{
	if (nalternatives == 0)
		throw compiler_error("Alternatives of a control type must be non-zero.");
}

std::string
ctltype::debug_string() const
{
	return detail::strfmt("ctl(", nalternatives_, ")");
}

bool
ctltype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const ctltype*>(&other);
	return type && type->nalternatives_ == nalternatives_;
}

std::unique_ptr<jive::type>
ctltype::copy() const
{
	return std::unique_ptr<jive::type>(new ctltype(*this));
}

const ctltype ctl2(2);

/* control value representation */

ctlvalue_repr::ctlvalue_repr(size_t alternative, size_t nalternatives)
: alternative_(alternative)
, nalternatives_(nalternatives)
{
	if (alternative >= nalternatives)
		throw compiler_error("Alternative is bigger than the number of possible alternatives.");
}

/* match operator */

match_op::~match_op() noexcept
{}

match_op::match_op(
	size_t nbits,
	const std::unordered_map<uint64_t, uint64_t> & mapping,
	uint64_t default_alternative,
	size_t nalternatives)
: jive::unary_op(bittype(nbits), ctltype(nalternatives))
, default_alternative_(default_alternative)
, mapping_(mapping)
{}

bool
match_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const match_op*>(&other);
	return op
	    && op->default_alternative_ == default_alternative_
	    && op->mapping_ == mapping_
	    && op->nbits() == nbits()
	    && op->nalternatives() == nalternatives();
}

jive_unop_reduction_path_t
match_op::can_reduce_operand(const jive::output * arg) const noexcept
{
	if (is<bitconstant_op>(producer(arg)))
		return jive_unop_reduction_constant;

	return jive_unop_reduction_none;
}

jive::output *
match_op::reduce_operand(jive_unop_reduction_path_t path, jive::output * arg) const
{
	if (path == jive_unop_reduction_constant) {
		auto op = static_cast<const bitconstant_op&>(producer(arg)->operation());
		return jive_control_constant(arg->region(), nalternatives(),
			alternative(op.value().to_uint()));
	}

	return nullptr;
}

std::string
match_op::debug_string() const
{
	std::string str("[");
	for (const auto & pair : mapping_)
		str += detail::strfmt(pair.first, " -> ", pair.second, ", ");
	str += detail::strfmt(default_alternative_, "]");

	return "MATCH" + str;
}

std::unique_ptr<jive::operation>
match_op::copy() const
{
	return std::unique_ptr<jive::operation>(new match_op(*this));
}

jive::output *
match(
	size_t nbits,
	const std::unordered_map<uint64_t, uint64_t> & mapping,
	uint64_t default_alternative,
	size_t nalternatives,
	jive::output * operand)
{
	match_op op(nbits, mapping, default_alternative, nalternatives);
	return simple_node::create_normalized(operand->region(), op, {operand})[0];
}

}

jive::output *
jive_control_constant(jive::region * region, size_t nalternatives, size_t alternative)
{
	jive::ctlconstant_op op({alternative, nalternatives});
	return jive::simple_node::create_normalized(region, op, {})[0];
}
