/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_EXPRESSION_HPP
#define JLM_IR_EXPRESSION_HPP

#include <jlm/common.hpp>

#include <jive/rvsdg/operation.h>

#include <memory>
#include <vector>

namespace jlm {

class expr final {
public:
	inline
	~expr() noexcept
	{}

	inline
	expr(
		const jive::operation & operation,
		std::vector<std::unique_ptr<const expr>> operands)
		: operation_(std::move(operation.copy()))
		, operands_(std::move(operands))
	{
		/* FIXME: check number of results of operation */
	}

	expr(const expr &) = delete;

	inline
	expr(expr && other)
	: operation_(std::move(other.operation_))
	, operands_(std::move(other.operands_))
	{}

	expr &
	operator=(const expr &) = delete;

	inline expr &
	operator=(expr && other)
	{
		if (this == &other)
			return *this;

		operation_ = std::move(other.operation_);
		operands_ = std::move(other.operands_);
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

	const jive::type &
	type() const noexcept
	{
		return operation_->result(0).type();
	}

	inline std::string
	debug_string() const
	{
		/* FIXME: what about operands */
		return operation_->debug_string();
	}

private:
	std::unique_ptr<jive::operation> operation_;
	std::vector<std::unique_ptr<const expr>> operands_;
};

}

#endif
