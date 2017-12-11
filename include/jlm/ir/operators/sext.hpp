/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_SEXT_HPP
#define JLM_IR_OPERATORS_SEXT_HPP

#include <jive/types/bitstring.h>
#include <jive/rvsdg/unary.h>

#include <jlm/ir/tac.hpp>

namespace jlm {

/* sext operator */

class sext_op final : public jive::unary_op {
public:
	virtual
	~sext_op();

	inline
	sext_op(const jive::bits::type & otype, const jive::bits::type & rtype)
	: jive::unary_op()
	, oport_(otype)
	, rport_(rtype)
	{
		if (otype.nbits() >= rtype.nbits())
			throw std::logic_error("Expected operand's #bits to be smaller than results's #bits.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

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
		return static_cast<const jive::bits::type*>(&oport_.type())->nbits();
	}

	inline size_t
	ndstbits() const noexcept
	{
		return static_cast<const jive::bits::type*>(&rport_.type())->nbits();
	}

private:
	jive::port oport_;
	jive::port rport_;
};

static inline bool
is_sext_op(const jive::operation & op)
{
	return dynamic_cast<const sext_op*>(&op) != nullptr;
}

static inline bool
is_sext_node(const jive::node * node) noexcept
{
	return jive::is_opnode<sext_op>(node);
}

static inline std::unique_ptr<jlm::tac>
create_sext_tac(const variable * operand, jlm::variable * result)
{
	auto ot = dynamic_cast<const jive::bits::type*>(&operand->type());
	if (!ot) throw std::logic_error("Expected bits type.");

	auto rt = dynamic_cast<const jive::bits::type*>(&result->type());
	if (!rt) throw std::logic_error("Expected bits type.");

	sext_op op(*ot, *rt);
	return create_tac(op, {operand}, {result});
}

static inline jive::output *
create_sext(size_t ndstbits, jive::output * operand)
{
	auto ot = dynamic_cast<const jive::bits::type*>(&operand->type());
	if (!ot) throw std::logic_error("Expected bits type.");

	sext_op op(*ot, jive::bits::type(ndstbits));
	return jive::create_normalized(operand->region(), op, {operand})[0];
}

}

#endif
