/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */
#ifndef JLM_IR_CFG_H
#define JLM_IR_CFG_H

#include <jlm/common.hpp>
#include <jlm/jlm/ir/cfg-node.hpp>
#include <jlm/jlm/ir/variable.hpp>

#include <jive/rvsdg/operation.h>

namespace jive {
namespace base {
	class type;
}
}

namespace jlm {

class clg_node;
class basic_block;
class module;
class tac;

class entry final : public attribute {
public:
	virtual
	~entry();

	inline
	entry()
	: attribute()
	{}

	size_t
	narguments() const noexcept
	{
		return arguments_.size();
	}

	const variable *
	argument(size_t index) const
	{
		JLM_DEBUG_ASSERT(index < narguments());
		return arguments_[index];
	}

	inline void
	append_argument(const variable * v)
	{
		return arguments_.push_back(v);
	}

private:
	std::vector<const variable*> arguments_;
};

static inline bool
is_entry(const jlm::attribute & attribute) noexcept
{
	return dynamic_cast<const jlm::entry*>(&attribute) != nullptr;
}

static inline bool
is_entry_node(const jlm::cfg_node * node)
{
	return dynamic_cast<const jlm::entry*>(&node->attribute()) != nullptr;
}

class exit final : public attribute {
public:
	virtual
	~exit();

	inline
	exit()
	: attribute()
	{}

	size_t
	nresults() const noexcept
	{
		return results_.size();
	}

	const variable *
	result(size_t index) const
	{
		JLM_DEBUG_ASSERT(index < nresults());
		return results_[index];
	}

	inline void
	append_result(const variable * v)
	{
		results_.push_back(v);
	}

private:
	std::vector<const variable*> results_;
};

static inline bool
is_exit(const jlm::attribute & attribute) noexcept
{
	return dynamic_cast<const jlm::exit*>(&attribute) != nullptr;
}

static inline bool
is_exit_node(const jlm::cfg_node * node)
{
	return dynamic_cast<const jlm::exit*>(&node->attribute()) != nullptr;
}

class cfg final {
	class iterator final {
	public:
		inline
		iterator(std::unordered_set<std::unique_ptr<cfg_node>>::iterator it)
		: it_(it)
		{}

		inline bool
		operator==(const iterator & other) const noexcept
		{
			return it_ == other.it_;
		}

		inline bool
		operator!=(const iterator & other) const noexcept
		{
			return !(*this == other);
		}

		inline const iterator &
		operator++() noexcept
		{
			++it_;
			return *this;
		}

		inline const iterator
		operator++(int) noexcept
		{
			iterator tmp(it_);
			it_++;
			return tmp;
		}

		inline cfg_node *
		node() const noexcept
		{
			return it_->get();
		}

		inline cfg_node &
		operator*() const noexcept
		{
			return *it_->get();
		}

		inline cfg_node *
		operator->() const noexcept
		{
			return node();
		}

	private:
		std::unordered_set<std::unique_ptr<cfg_node>>::iterator it_;
	};

	class const_iterator final {
	public:
		inline
		const_iterator(std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it)
		: it_(it)
		{}

		inline bool
		operator==(const const_iterator & other) const noexcept
		{
			return it_ == other.it_;
		}

		inline bool
		operator!=(const const_iterator & other) const noexcept
		{
			return !(*this == other);
		}

		inline const const_iterator &
		operator++() noexcept
		{
			++it_;
			return *this;
		}

		inline const const_iterator
		operator++(int) noexcept
		{
			const_iterator tmp(it_);
			it_++;
			return tmp;
		}

		inline const cfg_node &
		operator*() noexcept
		{
			return *it_->get();
		}

		inline const cfg_node *
		operator->() noexcept
		{
			return it_->get();
		}

	private:
		std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it_;
	};

public:
	~cfg() {}

	cfg(jlm::module & module);

public:
	inline const_iterator
	begin() const
	{
		return const_iterator(nodes_.begin());
	}

	inline iterator
	begin()
	{
		return iterator(nodes_.begin());
	}

	inline const_iterator
	end() const
	{
		return const_iterator(nodes_.end());
	}

	inline iterator
	end()
	{
		return iterator(nodes_.end());
	}

	inline jlm::cfg_node *
	entry_node() const noexcept
	{
		return entry_;
	}

	inline jlm::entry &
	entry() const noexcept
	{
		return *static_cast<jlm::entry*>(&entry_node()->attribute());
	}

	inline jlm::cfg_node *
	exit_node() const noexcept
	{
		return exit_;
	}

	inline jlm::exit &
	exit() const noexcept
	{
		return *static_cast<jlm::exit*>(&exit_node()->attribute());
	}

	inline void
	add_node(std::unique_ptr<jlm::cfg_node> node)
	{
		nodes_.insert(std::move(node));
	}

	inline cfg::iterator
	find_node(jlm::cfg_node * n)
	{
		std::unique_ptr<cfg_node> up(n);
		auto it = nodes_.find(up);
		up.release();
		return iterator(it);
	}

	cfg::iterator
	remove_node(cfg::iterator & it);

	inline cfg::iterator
	remove_node(jlm::cfg_node * n)
	{
		auto it = find_node(n);
		if (it == end())
			throw jlm::error("node does not belong to this CFG.");

		return remove_node(it);
	}

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	inline jlm::module &
	module() const noexcept
	{
		return module_;
	}

private:
	cfg_node * entry_;
	cfg_node * exit_;
	jlm::module & module_;
	std::unordered_set<std::unique_ptr<cfg_node>> nodes_;
};

}

#endif
