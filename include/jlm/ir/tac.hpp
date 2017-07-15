/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TAC_H
#define JLM_IR_TAC_H

#include <jlm/common.hpp>
#include <jlm/ir/variable.hpp>

#include <jive/vsdg/operators/operation.h>

#include <memory>
#include <vector>

namespace jive {

namespace base {
	class type;
}
}

namespace jlm {

class tac;

/* tacvariable */

class tacvariable final : public variable {
public:
	virtual
	~tacvariable() noexcept;

	inline
	tacvariable(
		const jive::base::type & type,
		const std::string & name)
	: variable (type, name, false)
	, tac_(nullptr)
	{}

	inline jlm::tac *
	tac() const noexcept
	{
		return tac_;
	}

	/*
		FIXME: ensure tac is set in the constructor
	*/

	inline void
	set_tac(jlm::tac * tac) noexcept
	{
		JLM_DEBUG_ASSERT(tac_ == nullptr);
		tac_ = tac;
	}

private:
	jlm::tac * tac_;
};

static inline bool
is_tacvariable(const jlm::variable * v)
{
	return dynamic_cast<const jlm::tacvariable*>(v) != nullptr;
}

static inline std::unique_ptr<variable>
create_tacvariable(
	const jive::base::type & type,
	const std::string & name)
{
	return std::unique_ptr<variable>(new tacvariable(type, name));
}

/* tac */

class tac final {
public:
	inline
	~tac() noexcept
	{}

	tac(const jive::operation & operation,
		const std::vector<const variable*> & operands,
		const std::vector<const variable*> & results);

	tac(const jlm::tac &) = delete;

	tac(jlm::tac &&) = delete;

	tac &
	operator=(const jlm::tac &) = delete;

	tac &
	operator=(jlm::tac &&) = delete;

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

private:
	std::vector<const variable*> inputs_;
	std::vector<const variable*> outputs_;
	std::unique_ptr<jive::operation> operation_;
};

static inline std::unique_ptr<jlm::tac>
create_tac(
	const jive::operation & operation,
	const std::vector<const variable *> & operands,
	const std::vector<const variable *> & results)
{
	return std::make_unique<jlm::tac>(operation, operands, results);
}

}

#endif
