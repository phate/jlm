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

class basic_block final : public attribute {
public:
	virtual
	~basic_block();

	inline
	basic_block()
	: attribute()
	{}

	inline
	basic_block(const basic_block & other)
	: tacs_(other.tacs_)
	{}

	basic_block(basic_block &&) = delete;

	inline basic_block &
	operator=(const basic_block & other)
	{
		if (this == &other)
			return *this;

		tacs_ = other.tacs_;
	}

	basic_block &
	operator=(basic_block &&) = delete;

	const tac *
	append(
		const jive::operation & operation,
		const std::vector<const variable*> & operands,
		const std::vector<const variable*> & results);

	const variable *
	append(jlm::cfg * cfg, const expr & e, const variable * v);

	const variable *
	append(jlm::cfg * cfg, const expr & e);

	inline size_t
	ntacs() const noexcept
	{
		return tacs_.size();
	}

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

static inline cfg_node *
create_basic_block_node(jlm::cfg * cfg)
{
	basic_block attr;
	return cfg->create_node(attr);
}

static inline basic_block *
create_basic_block(jlm::cfg * cfg)
{
	return static_cast<basic_block*>(&create_basic_block_node(cfg)->attribute());
}

static inline bool
is_basic_block(const jlm::cfg_node * node) noexcept
{
	return dynamic_cast<const basic_block*>(&node->attribute()) != nullptr;
}

}

#endif
