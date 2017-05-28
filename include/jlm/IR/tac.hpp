/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TAC_H
#define JLM_IR_TAC_H

#include <jlm/common.hpp>
#include <jlm/IR/variable.hpp>

#include <jive/vsdg/operators/operation.h>

#include <memory>
#include <vector>

namespace jive {

namespace base {
	class type;
}
}

namespace jlm {

class variable;

class tac final {
public:
	inline
	~tac() noexcept
	{}

	tac(const jive::operation & operation,
		const std::vector<const variable*> & operands,
		const std::vector<const variable*> & results);

	inline
	tac(const jlm::tac & other)
	: inputs_(other.inputs_)
	, outputs_(other.outputs_)
	, operation_(std::move(other.operation().copy()))
	{}

	tac &
	operator=(const jlm::tac & other)
	{
		if (this == &other)
			return *this;

		inputs_ = other.inputs_;
		outputs_ = other.outputs_;
		operation_ = std::move(other.operation().copy());

		return *this;
	}

	inline const jive::operation &
	operation() const noexcept
	{
		return *operation_;
	}

	inline size_t
	ninputs() const noexcept
	{
		return inputs_.size();
	}

	inline const variable *
	input(size_t index) const noexcept
	{
		JLM_DEBUG_ASSERT(index < inputs_.size());
		return inputs_[index];
	}

	inline size_t
	noutputs() const noexcept
	{
		return outputs_.size();
	}

	inline const variable *
	output(size_t index) const noexcept
	{
		JLM_DEBUG_ASSERT(index < outputs_.size());
		return outputs_[index];
	}

	std::string
	debug_string() const;

private:
	std::vector<const variable*> inputs_;
	std::vector<const variable*> outputs_;
	std::unique_ptr<jive::operation> operation_;
};

static inline std::unique_ptr<jlm::tac>
create_tac(
	const jive::operation & operation,
	const std::vector<const variable*> & operands,
	const std::vector<const variable*> & results)
{
	return std::make_unique<jlm::tac>(operation, operands, results);
}

}

#endif
