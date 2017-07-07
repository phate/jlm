/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_BASIC_BLOCK_H
#define JLM_IR_BASIC_BLOCK_H

#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg_node.hpp>
#include <jlm/ir/tac.hpp>

#include <list>

namespace jive {
	class operation;
}

namespace jlm {

class expr;

class basic_block final : public attribute {
public:
	typedef std::list<const tac*>::const_iterator const_iterator;
	typedef std::list<const tac*>::const_reverse_iterator const_reverse_iterator;

	virtual
	~basic_block();

	inline
	basic_block()
	: attribute()
	{}

	basic_block(const basic_block&) = delete;

	basic_block(basic_block && other)
	: attribute(other)
	, tacs_(std::move(other.tacs_))
	{}

	basic_block &
	operator=(const basic_block &) = delete;

	basic_block &
	operator=(basic_block && other)
	{
		if (this == &other)
			return *this;

		for (const auto & tac : other.tacs_)
			delete tac;

		tacs_.clear();
		tacs_ = std::move(other.tacs_);

		return *this;
	}

	inline const_iterator
	begin() const noexcept
	{
		return tacs_.begin();
	}

	inline const_reverse_iterator
	rbegin() const noexcept
	{
		return tacs_.rbegin();
	}

	inline const_iterator
	end() const noexcept
	{
		return tacs_.end();
	}

	inline const_reverse_iterator
	rend() const noexcept
	{
		return tacs_.rend();
	}

	inline const tac *
	insert(const const_iterator & it, std::unique_ptr<jlm::tac> tac)
	{
		return *tacs_.insert(it, tac.release());
	}

	inline void
	insert(const const_iterator & it, std::vector<std::unique_ptr<jlm::tac>> & tacs)
	{
		for (auto & tac : tacs)
			tacs_.insert(it, tac.release());
	}

	inline const tac *
	append_last(std::unique_ptr<jlm::tac> tac)
	{
		tacs_.push_back(tac.release());
		return tacs_.back();
	}

	inline void
	append_last(std::vector<std::unique_ptr<jlm::tac>> & tacs)
	{
		for (auto & tac : tacs)
			tacs_.push_back(tac.release());
	}

	inline const tac *
	append_first(std::unique_ptr<jlm::tac> tac)
	{
		tacs_.push_front(tac.release());
		return tacs_.front();
	}

	inline void
	append_first(std::vector<std::unique_ptr<jlm::tac>> & tacs)
	{
		for (auto & tac : tacs)
			tacs_.push_front(tac.release());
	}

	inline void
	append_first(basic_block & bb)
	{
		tacs_.insert(tacs_.begin(), bb.begin(), bb.end());
		bb.tacs_.clear();
	}

	inline size_t
	ntacs() const noexcept
	{
		return tacs_.size();
	}

	inline const tac *
	first() const noexcept
	{
		return ntacs() != 0 ? tacs_.front() : nullptr;
	}

	inline const tac *
	last() const noexcept
	{
		return ntacs() != 0 ? tacs_.back() : nullptr;
	}

	inline void
	drop_first()
	{
		tacs_.pop_front();
	}

	inline void
	drop_last()
	{
		tacs_.pop_back();
	}

private:
	std::list<const tac*> tacs_;
};

static inline bool
is_basic_block(const jlm::attribute & attribute) noexcept
{
	return dynamic_cast<const jlm::basic_block*>(&attribute) != nullptr;
}

static inline cfg_node *
create_basic_block_node(jlm::cfg * cfg)
{
	return cfg_node::create(*cfg, std::make_unique<basic_block>());
}

static inline basic_block *
create_basic_block(jlm::cfg * cfg)
{
	return static_cast<basic_block*>(&create_basic_block_node(cfg)->attribute());
}

static inline const jlm::tac *
insert(
	jlm::cfg_node * node,
	const basic_block::const_iterator & it,
	std::unique_ptr<jlm::tac> tac)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());
	return bb.insert(it, std::move(tac));
}

static inline void
insert(
	jlm::cfg_node * node,
	const basic_block::const_iterator & it,
	std::vector<std::unique_ptr<jlm::tac>> & tacs)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());
	return bb.insert(it, tacs);
}

static inline const jlm::tac *
append_last(jlm::cfg_node * node, std::unique_ptr<jlm::tac> tac)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());
	bb.append_last(std::move(tac));
	return bb.last();
}

static inline void
append_last(jlm::cfg_node * node, std::vector<std::unique_ptr<jlm::tac>> & tacs)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());
	bb.append_last(tacs);
}

static inline const jlm::tac *
append_first(jlm::cfg_node * node, std::unique_ptr<jlm::tac> tac)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());
	bb.append_first(std::move(tac));
	return bb.first();
}

static inline void
append_first(jlm::cfg_node * node, std::vector<std::unique_ptr<jlm::tac>> & tacs)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());
	bb.append_first(tacs);
}

static inline void
append_first(jlm::cfg_node * node, jlm::cfg_node * source)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()) && is_basic_block(source->attribute()));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());
	auto & s = *static_cast<jlm::basic_block*>(&source->attribute());
	bb.append_first(s);
}

}

#endif
