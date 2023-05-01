/*
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_COMPARISON_HPP
#define JLM_RVSDG_BITSTRING_COMPARISON_HPP

#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jive {

#define DECLARE_BITCOMPARISON_OPERATION(NAME) \
class NAME ## _op final : public bitcompare_op { \
public: \
	virtual \
	~NAME ## _op() noexcept; \
\
	inline \
	NAME ## _op(const bittype & type) noexcept \
	: bitcompare_op(type) \
	{} \
\
	virtual bool \
	operator==(const operation & other) const noexcept override; \
\
	virtual enum jive::binary_op::flags \
	flags() const noexcept override; \
\
	virtual compare_result \
	reduce_constants( \
		const bitvalue_repr & arg1, \
		const bitvalue_repr & arg2) const override; \
\
	virtual std::string \
	debug_string() const override; \
\
	virtual std::unique_ptr<jive::operation> \
	copy() const override; \
\
	virtual std::unique_ptr<bitcompare_op> \
	create(size_t nbits) const override; \
\
	static inline jive::output * \
	create(size_t nbits, jive::output * op1, jive::output * op2) \
	{ \
		return simple_node::create_normalized(op1->region(), NAME ## _op(nbits), {op1, op2})[0]; \
	} \
}; \

DECLARE_BITCOMPARISON_OPERATION(biteq)
DECLARE_BITCOMPARISON_OPERATION(bitne)
DECLARE_BITCOMPARISON_OPERATION(bitsge)
DECLARE_BITCOMPARISON_OPERATION(bitsgt)
DECLARE_BITCOMPARISON_OPERATION(bitsle)
DECLARE_BITCOMPARISON_OPERATION(bitslt)
DECLARE_BITCOMPARISON_OPERATION(bituge)
DECLARE_BITCOMPARISON_OPERATION(bitugt)
DECLARE_BITCOMPARISON_OPERATION(bitule)
DECLARE_BITCOMPARISON_OPERATION(bitult)

}

#endif
