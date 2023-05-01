/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/concat.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/slice.hpp>

namespace jive {

bitslice_op::~bitslice_op() noexcept
{}

bool
bitslice_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const bitslice_op*>(&other);
	return op
	    && op->low() == low()
	    && op->high() == high()
	    && op->argument(0) == argument(0);
}

std::string
bitslice_op::debug_string() const
{
	return detail::strfmt("SLICE[", low(), ":", high(), ")");
}

jive_unop_reduction_path_t
bitslice_op::can_reduce_operand(const jive::output * arg) const noexcept
{
	auto node = node_output::node(arg);
	auto & arg_type = *dynamic_cast<const bittype*>(&arg->type());

	if ((low() == 0) && (high() == arg_type.nbits()))
		return jive_unop_reduction_idempotent;

	if (is<bitslice_op>(node))
		return jive_unop_reduction_narrow;

	if (is<bitconstant_op>(node))
		return jive_unop_reduction_constant;

	if (is<bitconcat_op>(node))
		return jive_unop_reduction_distribute;
	
	return jive_unop_reduction_none;
}

jive::output *
bitslice_op::reduce_operand(
	jive_unop_reduction_path_t path,
	jive::output * arg) const
{
	if (path == jive_unop_reduction_idempotent) {
		return arg;
	}

	auto node = static_cast<node_output*>(arg)->node();
	
	if (path == jive_unop_reduction_narrow) {
		auto op = static_cast<const bitslice_op&>(node->operation());
		return jive_bitslice(node->input(0)->origin(), low() + op.low(), high() + op.low());
	}
	
	if (path == jive_unop_reduction_constant) {
		auto op = static_cast<const bitconstant_op&>(node->operation());
		std::string s(&op.value()[0]+low(), high()-low());
		return create_bitconstant(arg->region(), s.c_str());
	}
	
	if (path == jive_unop_reduction_distribute) {
		size_t pos = 0, n;
		std::vector<jive::output*> arguments;
		for (n = 0; n < node->ninputs(); n++) {
			auto argument = node->input(n)->origin();
			size_t base = pos;
			size_t nbits = static_cast<const bittype&>(argument->type()).nbits();
			pos = pos + nbits;
			if (base < high() && pos > low()) {
				size_t slice_low = (low() > base) ? (low() - base) : 0;
				size_t slice_high = (high() < pos) ? (high() - base) : (pos - base);
				argument = jive_bitslice(argument, slice_low, slice_high);
				arguments.push_back(argument);
			}
		}
		
		return jive_bitconcat(arguments);
	}
	
	return nullptr;
}

std::unique_ptr<jive::operation>
bitslice_op::copy() const
{
	return std::unique_ptr<jive::operation>(new bitslice_op(*this));
}

}

jive::output *
jive_bitslice(jive::output * argument, size_t low, size_t high)
{
	auto & type = dynamic_cast<const jive::bittype&>(argument->type());
	jive::bitslice_op op(type, low, high);
	return jive::simple_node::create_normalized(argument->region(), op, {argument})[0];
}
