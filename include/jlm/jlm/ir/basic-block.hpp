/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_BASIC_BLOCK_H
#define JLM_IR_BASIC_BLOCK_H

#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/cfg-node.hpp>
#include <jlm/jlm/ir/tac.hpp>

#include <list>

namespace jive {
	class operation;
}

namespace jlm {

class expr;

class taclist final : public attribute {
public:
	typedef std::list<tac*>::const_iterator const_iterator;
	typedef std::list<tac*>::const_reverse_iterator const_reverse_iterator;

	virtual
	~taclist();

	inline
	taclist()
	: attribute()
	{}

	taclist(const taclist&) = delete;

	taclist(taclist && other)
	: attribute(other)
	, tacs_(std::move(other.tacs_))
	{}

	taclist &
	operator=(const taclist &) = delete;

	taclist &
	operator=(taclist && other)
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

	inline tac *
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

	inline tac *
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

	inline tac *
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
	append_first(taclist & bb)
	{
		tacs_.insert(tacs_.begin(), bb.begin(), bb.end());
		bb.tacs_.clear();
	}

	inline size_t
	ntacs() const noexcept
	{
		return tacs_.size();
	}

	inline tac *
	first() const noexcept
	{
		return ntacs() != 0 ? tacs_.front() : nullptr;
	}

	inline tac *
	last() const noexcept
	{
		return ntacs() != 0 ? tacs_.back() : nullptr;
	}

	inline void
	drop_first()
	{
		delete tacs_.front();
		tacs_.pop_front();
	}

	inline void
	drop_last()
	{
		delete tacs_.back();
		tacs_.pop_back();
	}

private:
	std::list<tac*> tacs_;
};

/* basic block */

class basic_block final : public cfg_node {
public:
	virtual
	~basic_block();

private:
	basic_block(jlm::cfg & cfg)
	: cfg_node(cfg, nullptr)
	{}

	basic_block(const basic_block&) = delete;

	basic_block(basic_block &&) = delete;

	basic_block &
	operator=(const basic_block&) = delete;

	basic_block &
	operator=(basic_block&&) = delete;

public:
	const taclist &
	tacs() const noexcept
	{
		return tacs_;
	}

	taclist &
	tacs() noexcept
	{
		return tacs_;
	}

	static basic_block *
	create(jlm::cfg & cfg);

private:
	taclist tacs_;
};

static inline bool
is_basic_block(const cfg_node * node) noexcept
{
	return dynamic_cast<const basic_block*>(node) != nullptr;
}

static inline jlm::tac *
insert(
	jlm::cfg_node * node,
	const taclist::const_iterator & it,
	std::unique_ptr<jlm::tac> tac)
{
	JLM_DEBUG_ASSERT(is_basic_block(node));
	auto bb = static_cast<basic_block*>(node);
	return bb->tacs().insert(it, std::move(tac));
}

static inline void
insert(
	jlm::cfg_node * node,
	const taclist::const_iterator & it,
	std::vector<std::unique_ptr<jlm::tac>> & tacs)
{
	JLM_DEBUG_ASSERT(is_basic_block(node));
	auto bb = static_cast<basic_block*>(node);
	return bb->tacs().insert(it, tacs);
}

static inline jlm::tac *
append_last(jlm::cfg_node * node, std::unique_ptr<jlm::tac> tac)
{
	JLM_DEBUG_ASSERT(is_basic_block(node));
	auto bb = static_cast<basic_block*>(node);
	bb->tacs().append_last(std::move(tac));
	return bb->tacs().last();
}

static inline void
append_last(jlm::cfg_node * node, std::vector<std::unique_ptr<jlm::tac>> & tacs)
{
	JLM_DEBUG_ASSERT(is_basic_block(node));
	auto bb = static_cast<basic_block*>(node);
	bb->tacs().append_last(tacs);
}

static inline jlm::tac *
append_first(jlm::cfg_node * node, std::unique_ptr<jlm::tac> tac)
{
	JLM_DEBUG_ASSERT(is_basic_block(node));
	auto bb = static_cast<basic_block*>(node);
	bb->tacs().append_first(std::move(tac));
	return bb->tacs().first();
}

static inline void
append_first(jlm::cfg_node * node, std::vector<std::unique_ptr<jlm::tac>> & tacs)
{
	JLM_DEBUG_ASSERT(is_basic_block(node));
	auto bb = static_cast<basic_block*>(node);
	bb->tacs().append_first(tacs);
}

static inline void
append_first(jlm::cfg_node * node, jlm::cfg_node * source)
{
	JLM_DEBUG_ASSERT(is_basic_block(node) && is_basic_block(source));
	auto bb = static_cast<basic_block*>(node);
	auto s = static_cast<basic_block*>(source);
	bb->tacs().append_first(s->tacs());
}

}

#endif
