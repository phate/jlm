/*
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_ARITHMETIC_HPP
#define JLM_RVSDG_BITSTRING_ARITHMETIC_HPP

#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jive {

#define DECLARE_BITUNARY_OPERATION(NAME) \
class NAME ## _op final : public bitunary_op { \
public: \
	virtual \
	~NAME ## _op() noexcept; \
\
	inline \
	NAME ## _op(const bittype & type) noexcept \
	: bitunary_op(type) \
	{} \
\
	virtual bool \
	operator==(const operation & other) const noexcept override; \
\
	virtual bitvalue_repr \
	reduce_constant(const bitvalue_repr & arg) const override; \
\
	virtual std::string \
	debug_string() const override; \
\
	virtual std::unique_ptr<jive::operation> \
	copy() const override; \
\
	virtual std::unique_ptr<bitunary_op> \
	create(size_t nbits) const override; \
\
	static inline jive::output * \
	create(size_t nbits, jive::output * op) \
	{ \
		return simple_node::create_normalized(op->region(), NAME ## _op(nbits), {op})[0]; \
	} \
}; \

#define DECLARE_BITBINARY_OPERATION(NAME) \
class NAME ## _op final : public bitbinary_op { \
public: \
	virtual \
	~NAME ## _op() noexcept; \
\
	inline \
	NAME ## _op(const bittype & type) noexcept \
	: bitbinary_op(type) \
	{} \
\
	virtual bool \
	operator==(const operation & other) const noexcept override; \
\
	virtual enum jive::binary_op::flags \
	flags() const noexcept override; \
\
	virtual bitvalue_repr \
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
	virtual std::unique_ptr<bitbinary_op> \
	create(size_t nbits) const override; \
\
	static inline jive::output * \
	create(size_t nbits, jive::output * op1, jive::output * op2) \
	{ \
		return simple_node::create_normalized(op1->region(), NAME ## _op(nbits), {op1, op2})[0]; \
	} \
}; \

DECLARE_BITUNARY_OPERATION(bitneg)
DECLARE_BITUNARY_OPERATION(bitnot)

DECLARE_BITBINARY_OPERATION(bitadd)
DECLARE_BITBINARY_OPERATION(bitand)
DECLARE_BITBINARY_OPERATION(bitashr)
DECLARE_BITBINARY_OPERATION(bitmul)
DECLARE_BITBINARY_OPERATION(bitor)
DECLARE_BITBINARY_OPERATION(bitsdiv)
DECLARE_BITBINARY_OPERATION(bitshl)
DECLARE_BITBINARY_OPERATION(bitshr)
DECLARE_BITBINARY_OPERATION(bitsmod)
DECLARE_BITBINARY_OPERATION(bitsmulh)
DECLARE_BITBINARY_OPERATION(bitsub)
DECLARE_BITBINARY_OPERATION(bitudiv)
DECLARE_BITBINARY_OPERATION(bitumod)
DECLARE_BITBINARY_OPERATION(bitumulh)
DECLARE_BITBINARY_OPERATION(bitxor)

}

#endif
