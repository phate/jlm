/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_BASIC_BLOCK_H
#define JLM_IR_BASIC_BLOCK_H

#include <jlm/IR/cfg.hpp>
#include <jlm/IR/cfg_node.hpp>

namespace jive {
	class operation;
}

namespace jlm {

class expr;
class tac;

class basic_block_attribute final : public attribute {
public:
	virtual
	~basic_block_attribute();

	inline
	basic_block_attribute()
	: attribute()
	{}

	inline
	basic_block_attribute(const basic_block_attribute & other)
	: tacs_(other.tacs_)
	{}

	basic_block_attribute(basic_block_attribute &&) = delete;

	inline basic_block_attribute &
	operator=(const basic_block_attribute & other)
	{
		if (this == &other)
			return *this;

		tacs_ = other.tacs_;
	}

	basic_block_attribute &
	operator=(basic_block_attribute &&) = delete;

	const tac *
	append(
		const jive::operation & operation,
		const std::vector<const variable*> & operands,
		const std::vector<const variable*> & results);

	const tac *
	append(
		jlm::cfg * cfg,
		const jive::operation & operation,
		const std::vector<const variable*> & operands);

	const variable *
	append(jlm::cfg * cfg, const expr & e, const variable * v);

	const variable *
	append(jlm::cfg * cfg, const expr & e);

	/*
		FIXME: add accessor functions
	*/
	inline std::list<const tac*> &
	tacs() noexcept
	{
		return tacs_;
	}

	inline const std::list<const tac*> &
	tacs() const noexcept
	{
		return tacs_;
	}

	virtual std::string
	debug_string() const noexcept override;

	virtual std::unique_ptr<attribute>
	copy() const override;

private:
	std::list<const tac*> tacs_;
};

static inline basic_block_attribute &
create_basic_block(jlm::cfg & cfg)
{
	basic_block_attribute attr;
	return *static_cast<basic_block_attribute*>(&cfg.create_node(attr)->attribute());
}

class basic_block final : public cfg_node {
public:
	virtual ~basic_block();

	virtual std::string debug_string() const override;

	const tac *
	append(const jive::operation & operation, const std::vector<const variable*> & operands);

	const tac *
	append(const jive::operation & operation, const std::vector<const variable*> & operands,
		const std::vector<const variable*> & results);

	const variable *
	append(const expr & e, const variable * v);

	const variable *
	append(const expr & e);

	/*
		FIXME: add accessor functions
	*/
	inline std::list<const tac*> &
	tacs() noexcept
	{
		return tacs_;
	}

	inline const std::list<const tac*> &
	tacs() const noexcept
	{
		return tacs_;
	}

private:
	basic_block(jlm::cfg & cfg) noexcept;

	std::list<const tac*> tacs_;

	friend basic_block * cfg::create_basic_block();
};

}

#endif
