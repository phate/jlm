/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_EXPRESSION_HPP
#define JLM_IR_EXPRESSION_HPP

#include <jlm/common.hpp>

#include <jive/vsdg/operators/base.h>

#include <memory>
#include <vector>

namespace jlm {

class expr final {
public:
	inline
	~expr() noexcept
	{}

	inline
	expr(const jive::operation & operation,
		const std::vector<std::shared_ptr<const expr>> & operands)
		: operation_(std::move(operation.copy()))
		, operands_(operands)
	{
		/* FIXME: check number of results of operation */
	}

	inline
	expr(const expr & other)
		: operation_(std::move(other.operation_->copy()))
		, operands_(other.operands_)
	{}

	inline expr &
	operator=(const expr & other)
	{
		operation_ = std::move(other.operation_->copy());
		operands_ = other.operands_;

		return *this;
	}

	inline const jive::operation &
	operation() const noexcept
	{
		return *operation_;
	}

	inline size_t
	noperands() const noexcept
	{
		return operands_.size();
	}

	inline const expr &
	operand(size_t index) const noexcept
	{
		JLM_DEBUG_ASSERT(index < operands_.size());
		return *operands_[index];
	}

	const jive::base::type &
	type() const noexcept
	{
		return operation_->result_type(0);
	}

	inline std::string
	debug_string() const
	{
		/* FIXME: what about operands */
		return operation_->debug_string();
	}

private:
	std::unique_ptr<jive::operation> operation_;
	std::vector<std::shared_ptr<const expr>> operands_;
};

}

#endif
