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

/* cfg entry node */

class entry_node final : public cfg_node {
public:
	virtual
	~entry_node();

	entry_node(jlm::cfg & cfg)
	: cfg_node(cfg)
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

	const std::vector<const variable*> &
	arguments() const noexcept
	{
		return arguments_;
	}

	static entry_node *
	create(jlm::cfg & cfg);

private:
	std::vector<const variable*> arguments_;
};

/* cfg exit node */

class exit_node final : public cfg_node {
public:
	virtual
	~exit_node();

	exit_node(jlm::cfg & cfg)
	: cfg_node(cfg)
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

	const std::vector<const variable*>
	results() const noexcept
	{
		return results_;
	}

	static exit_node *
	create(jlm::cfg & cfg);

private:
	std::vector<const variable*> results_;
};

/* control flow graph */

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

	cfg(const cfg&) = delete;

	cfg(cfg&&) = delete;

	cfg &
	operator=(const cfg&) = delete;

	cfg &
	operator=(cfg&&) = delete;

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

	inline jlm::entry_node *
	entry() const noexcept
	{
		return entry_;
	}

	inline jlm::exit_node *
	exit_node() const noexcept
	{
		return exit_;
	}

	inline cfg_node *
	add_node(std::unique_ptr<jlm::cfg_node> node)
	{
		auto tmp = node.get();
		nodes_.insert(std::move(node));
		return tmp;
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
	jlm::entry_node * entry_;
	jlm::exit_node * exit_;
	jlm::module & module_;
	std::unordered_set<std::unique_ptr<cfg_node>> nodes_;
};

}

#endif
