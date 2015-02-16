/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TAC_TAC_H
#define JLM_IR_TAC_TAC_H

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
namespace frontend {

class input;
class output;
class variable;

class tac final {
public:
	~tac() noexcept;

	/*
		FIXME: to be removed
	*/
	tac(const cfg_node * owner, const jive::operation & operation,
		const std::vector<const output*> & operands);

	tac(const cfg_node * owner, const jive::operation & operation,
		const std::vector<const variable*> & operands,
		const std::vector<const variable*> & results);

	tac(const jlm::frontend::tac & tac) = delete;

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

	inline std::vector<const output*>
	outputs() const noexcept
	{
		return outputs_;
	}

	virtual std::string
	debug_string() const;

private:
	const cfg_node * owner_;
	std::vector<const variable*> inputs_;
	std::vector<const output*> outputs_;
	std::unique_ptr<jive::operation> operation_;
};

class input final {
public:
	input(const input &) = delete;

	input &
	operator=(const input &) = delete;

	inline const jive::base::type &
	type() const noexcept
	{
		return tac_->operation().argument_type(index_);
	}

	inline size_t
	index() const noexcept
	{
		return index_;
	}

	inline const output *
	origin() const noexcept
	{
		return origin_;
	}

	inline const jlm::frontend::variable *
	variable() const noexcept;

private:
	const jlm::frontend::tac * tac_;
	size_t index_;
	const output * origin_;

	input(const jlm::frontend::tac * tac, size_t index, const output * origin);

	friend jlm::frontend::tac::tac(const cfg_node * owner,
		const jive::operation & operation, const std::vector<const output*> & operands);

	friend jlm::frontend::tac::tac(const cfg_node * owner,
		const jive::operation & operation,
		const std::vector<const jlm::frontend::variable*> & operands,
		const std::vector<const jlm::frontend::variable*> & variables);
};

class output final {
public:
	output(const output &) = delete;

	output &
	operator=(const output &) = delete;

	inline const jive::base::type &
	type() const noexcept
	{
		return tac_->operation().result_type(index_);
	}

	inline size_t
	index() const noexcept
	{
		return index_;
	}

	inline const jlm::frontend::variable *
	variable() const noexcept
	{
		return variable_;
	}

private:
	const jlm::frontend::tac * tac_;
	size_t index_;
	const jlm::frontend::variable * variable_;

	output (const jlm::frontend::tac * tac, size_t index, const jlm::frontend::variable * variable);

	friend jlm::frontend::tac::tac(const cfg_node * owner,
		const jive::operation & operation, const std::vector<const output*> & operands);

	friend jlm::frontend::tac::tac(const cfg_node * owner,
		const jive::operation & operation,
		const std::vector<const jlm::frontend::variable*> & operands,
		const std::vector<const jlm::frontend::variable*> & variables);
};

inline const jlm::frontend::variable *
input::variable() const noexcept
{
	return origin()->variable();
}

}
}

#endif
