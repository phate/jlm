/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_AGGREGATION_STRUCTURE_HPP
#define JLM_IR_AGGREGATION_STRUCTURE_HPP

#include <jlm/IR/basic_block.hpp>

#include <string>

namespace jlm {
namespace agg {

/* structure */

class structure {
public:
	virtual
	~structure();

	inline constexpr
	structure()
	{}

	virtual std::string
	debug_string() const = 0;
};

/* entry */

class entry final : public structure {
public:
	virtual
	~entry();

	inline
	entry(const jlm::entry_attribute & attribute)
	: attribute_(attribute)
	{}

	inline const jlm::entry_attribute &
	attribute() const noexcept
	{
		return attribute_;
	}

	virtual std::string
	debug_string() const override;

private:
	jlm::entry_attribute attribute_;
};

static inline bool
is_entry_structure(const structure & s) noexcept
{
	return dynamic_cast<const entry*>(&s) != nullptr;
}

/* exit */

class exit final : public structure {
public:
	virtual
	~exit();

	inline
	exit(const jlm::exit_attribute & attribute)
	: attribute_(attribute)
	{}

	inline const jlm::exit_attribute &
	attribute() const noexcept
	{
		return attribute_;
	}

	virtual std::string
	debug_string() const override;

private:
	jlm::exit_attribute attribute_;
};

static inline bool
is_exit_structure(const structure & s) noexcept
{
	return dynamic_cast<const exit*>(&s) != nullptr;
}

/* block */

class block final : public structure {
public:
	virtual
	~block();

	inline
	block(const jlm::basic_block & bb)
	: bb_(bb)
	{}

	inline const jlm::basic_block &
	basic_block() const noexcept
	{
		return bb_;
	}

	virtual std::string
	debug_string() const override;

private:
	jlm::basic_block bb_;
};

static inline bool
is_block_structure(const structure & s) noexcept
{
	return dynamic_cast<const block*>(&s) != nullptr;
}

/* linear */

class linear final : public structure {
public:
	virtual
	~linear();

	inline constexpr
	linear()
	{}

	virtual std::string
	debug_string() const override;
};

static inline bool
is_linear_structure(const structure & s) noexcept
{
	return dynamic_cast<const linear*>(&s) != nullptr;
}

/* branch */

class branch final : public structure {
public:
	virtual
	~branch();

	inline
	branch()
	{}

	virtual std::string
	debug_string() const override;
};

static inline bool
is_branch_structure(const structure & s) noexcept
{
	return dynamic_cast<const branch*>(&s) != nullptr;
}

/* loop */

class loop final : public structure {
public:
	virtual
	~loop();

	inline constexpr
	loop()
	{}

	virtual std::string
	debug_string() const override;
};

static inline bool
is_loop_structure(const structure & s) noexcept
{
	return dynamic_cast<const loop*>(&s) != nullptr;
}

}}

#endif
