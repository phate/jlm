/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_AGGREGATION_NODE_HPP
#define JLM_IR_AGGREGATION_NODE_HPP

#include <jlm/ir/aggregation/structure.hpp>

#include <memory>
#include <vector>

namespace jlm {
namespace agg {

class node final {
	class iterator final {
	public:
		inline
		iterator(std::vector<std::unique_ptr<node>>::iterator it)
		: it_(std::move(it))
		{}

		inline const iterator &
		operator++() noexcept
		{
			it_++;
			return *this;
		}

		inline iterator
		operator++(int) noexcept
		{
			auto tmp = *this;
			it_++;
			return tmp;
		}

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

		inline node &
		operator*() const noexcept
		{
			return *it_->get();
		}

		inline node *
		operator->() const noexcept
		{
			return it_->get();
		}

	private:
		std::vector<std::unique_ptr<node>>::iterator it_;
	};

	class const_iterator final {
	public:
		inline
		const_iterator(std::vector<std::unique_ptr<node>>::const_iterator it)
		: it_(std::move(it))
		{}

		inline const const_iterator &
		operator++() noexcept
		{
			it_++;
			return *this;
		}

		inline const_iterator
		operator++(int) noexcept
		{
			auto tmp = *this;
			it_++;
			return tmp;
		}

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

		inline node &
		operator*() const noexcept
		{
			return *it_->get();
		}

		inline node *
		operator->() const noexcept
		{
			return it_->get();
		}

	private:
		std::vector<std::unique_ptr<node>>::const_iterator it_;
	};

public:
	inline
	~node()
	{}

	inline
	node(std::unique_ptr<jlm::agg::structure> structure)
	: parent_(nullptr)
	, structure_(std::move(structure))
	{}

	node(const node & other) = delete;

	node(node && other) = delete;

	node &
	operator=(const node & other) = delete;

	node &
	operator=(node && other) = delete;

	inline iterator
	begin() noexcept
	{
		return iterator(children_.begin());
	}

	inline const_iterator
	begin() const noexcept
	{
		return const_iterator(children_.begin());
	}

	inline iterator
	end() noexcept
	{
		return iterator(children_.end());
	}

	inline const_iterator
	end() const noexcept
	{
		return const_iterator(children_.end());
	}

	inline size_t
	nchildren() const noexcept
	{
		return children_.size();
	}

	inline void
	add_child(std::unique_ptr<node> child)
	{
		children_.emplace_back(std::move(child));
		children_[nchildren()-1]->parent_ = this;
	}

	inline node *
	child(size_t n) const noexcept
	{
		JLM_DEBUG_ASSERT(n < nchildren());
		return children_[n].get();
	}

	inline node *
	parent() noexcept
	{
		return parent_;
	}

	inline const jlm::agg::structure &
	structure() const noexcept
	{
		return *structure_;
	}

private:
	node * parent_;
	std::vector<std::unique_ptr<node>> children_;
	std::unique_ptr<jlm::agg::structure> structure_;
};

static inline std::unique_ptr<agg::node>
create_entry_node(const jlm::entry & attribute)
{
	return std::make_unique<agg::node>(std::make_unique<entry>(attribute));
}

static inline std::unique_ptr<agg::node>
create_exit_node(const jlm::exit & attribute)
{
	return std::make_unique<agg::node>(std::make_unique<exit>(attribute));
}

static inline std::unique_ptr<agg::node>
create_block_node(jlm::basic_block && bb)
{
	return std::make_unique<agg::node>(std::make_unique<block>(std::move(bb)));
}

static inline std::unique_ptr<agg::node>
create_linear_node(std::unique_ptr<agg::node> n1, std::unique_ptr<agg::node> n2)
{
	auto ln = std::make_unique<agg::node>(std::make_unique<linear>());
	ln->add_child(std::move(n1));
	ln->add_child(std::move(n2));
	return ln;
}

static inline std::unique_ptr<agg::node>
create_branch_node(std::unique_ptr<agg::node> split)
{
	auto b = std::make_unique<agg::node>(std::make_unique<branch>());
	b->add_child(std::move(split));
	return b;
}

static inline std::unique_ptr<agg::node>
create_loop_node(std::unique_ptr<agg::node> body)
{
	auto ln = std::make_unique<agg::node>(std::make_unique<loop>());
	ln->add_child(std::move(body));
	return ln;
}

}}

#endif
