/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_SEXT_HPP
#define JLM_LLVM_IR_OPERATORS_SEXT_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/unary.hpp>

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
			throw jlm::error("expected operand's #bits to be smaller than results's #bits.");
	}

	inline
	sext_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto ot = dynamic_cast<const jive::bittype*>(srctype.get());
		if (!ot) throw jlm::error("expected bits type.");

		auto rt = dynamic_cast<const jive::bittype*>(dsttype.get());
		if (!rt) throw jlm::error("expected bits type.");

		if (ot->nbits() >= rt->nbits())
			throw jlm::error("expected operand's #bits to be smaller than results's #bits.");
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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
		if (!ot) throw jlm::error("expected bits type.");

		auto rt = dynamic_cast<const jive::bittype*>(&type);
		if (!rt) throw jlm::error("expected bits type.");

		sext_op op(*ot, *rt);
		return tac::create(op, {operand});
	}

	static jive::output *
	create(
		size_t ndstbits,
		jive::output * operand)
	{
		auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
		if (!ot) throw jlm::error("expected bits type.");

		sext_op op(*ot, jive::bittype(ndstbits));
		return jive::simple_node::create_normalized(operand->region(), op, {operand})[0];
	}
};

}

#endif
