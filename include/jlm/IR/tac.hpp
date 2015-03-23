/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TAC_H
#define JLM_IR_TAC_H

#include <jlm/IR/basic_block.hpp>

#include <jive/vsdg/operators/base.h>

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

	tac(const cfg_node * owner, const jive::operation & operation,
		const std::vector<const variable*> & operands,
		const std::vector<const variable*> & results);

	tac(const jlm::tac & tac) = delete;

	tac &
	operator=(const tac &) = delete;

	inline const cfg_node *
	owner() const noexcept
	{
		return owner_;
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

	virtual std::string
	debug_string() const;

private:
	const cfg_node * owner_;
	std::vector<const variable*> inputs_;
	std::vector<const variable*> outputs_;
	std::unique_ptr<jive::operation> operation_;
};

}

#endif
