/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_SEXT_HPP
#define JLM_IR_OPERATORS_SEXT_HPP

#include <jive/types/bitstring.h>
#include <jive/rvsdg/unary.h>

#include <jlm/jlm/ir/tac.hpp>

namespace jlm {

/* sext operator */

class sext_op final : public jive::unary_op {
public:
	virtual
	~sext_op();

	inline
	sext_op(const jive::bittype & otype, const jive::bittype & rtype)
	: unary_op({otype}, {rtype})
	{
		if (otype.nbits() >= rtype.nbits())
			throw std::logic_error("Expected operand's #bits to be smaller than results's #bits.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	virtual jive_unop_reduction_path_t
	can_reduce_operand(const jive::output * operand) const noexcept override;

	virtual jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * operand) const override;

	inline size_t
	nsrcbits() const noexcept
	{
		return static_cast<const jive::bittype*>(&argument(0).type())->nbits();
	}

	inline size_t
	ndstbits() const noexcept
	{
		return static_cast<const jive::bittype*>(&result(0).type())->nbits();
	}
};

static inline std::unique_ptr<jlm::tac>
create_sext_tac(const variable * operand, jlm::variable * result)
{
	auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!ot) throw std::logic_error("Expected bits type.");

	auto rt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!rt) throw std::logic_error("Expected bits type.");

	sext_op op(*ot, *rt);
	return create_tac(op, {operand}, {result});
}

static inline jive::output *
create_sext(size_t ndstbits, jive::output * operand)
{
	auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!ot) throw std::logic_error("Expected bits type.");

	sext_op op(*ot, jive::bittype(ndstbits));
	return jive::simple_node::create_normalized(operand->region(), op, {operand})[0];
}

}

#endif
