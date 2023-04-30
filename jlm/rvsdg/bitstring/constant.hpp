/*
 * Copyright 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_CONSTANT_HPP
#define JLM_RVSDG_BITSTRING_CONSTANT_HPP

#include <stdint.h>
#include <vector>

#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/nullary.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jive {

struct type_of_value {
	bittype
	operator()(const bitvalue_repr & repr) const
	{
		return bittype(repr.nbits());
	}
};

struct format_value {
	std::string
	operator()(const bitvalue_repr & repr) const
	{
		if (repr.is_known() && repr.nbits() <= 64)
			return detail::strfmt("BITS", repr.nbits(), "(", repr.to_uint(), ")");

		return repr.str();
	}
};

typedef domain_const_op<bittype, bitvalue_repr, format_value, type_of_value> bitconstant_op;

inline bitconstant_op
uint_constant_op(size_t nbits, uint64_t value)
{
	return bitconstant_op(bitvalue_repr(nbits, value));
}

inline bitconstant_op
int_constant_op(size_t nbits, int64_t value)
{
	return bitconstant_op(bitvalue_repr(nbits, value));
}

// declare explicit instantiation
extern template class domain_const_op<bittype, bitvalue_repr, format_value, type_of_value>;

static inline jive::output *
create_bitconstant(jive::region * region, const bitvalue_repr & vr)
{
	return simple_node::create_normalized(region, bitconstant_op(vr), {})[0];
}

static inline jive::output *
create_bitconstant(jive::region * region, size_t nbits, int64_t value)
{
	return create_bitconstant(region, {nbits, value});
}

static inline jive::output *
create_bitconstant_undefined(jive::region * region, size_t nbits)
{
	std::string s(nbits, 'X');
	return create_bitconstant(region, bitvalue_repr(s.c_str()));
}

static inline jive::output *
create_bitconstant_defined(jive::region * region, size_t nbits)
{
	std::string s(nbits, 'D');
	return create_bitconstant(region, bitvalue_repr(s.c_str()));
}

}

#endif
