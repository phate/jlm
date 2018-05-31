/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_AGGREGATION_NODE_HPP
#define JLM_IR_AGGREGATION_NODE_HPP

#include <jlm/jlm/ir/aggregation/structure.hpp>

#include <memory>
#include <vector>

namespace jlm {
namespace agg {

class aggnode {
	class iterator final {
	public:
		inline
		iterator(std::vector<std::unique_ptr<aggnode>>::iterator it)
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

		inline aggnode &
		operator*() const noexcept
		{
			return *it_->get();
		}

		inline aggnode *
		operator->() const noexcept
		{
			return it_->get();
		}

	private:
		std::vector<std::unique_ptr<aggnode>>::iterator it_;
	};

	class const_iterator final {
	public:
		inline
		const_iterator(std::vector<std::unique_ptr<aggnode>>::const_iterator it)
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

		inline aggnode &
		operator*() const noexcept
		{
			return *it_->get();
		}

		inline aggnode *
		operator->() const noexcept
		{
			return it_->get();
		}

	private:
		std::vector<std::unique_ptr<aggnode>>::const_iterator it_;
	};

public:
	virtual
	~aggnode();

	inline
	aggnode(std::unique_ptr<jlm::agg::structure> structure)
	: parent_(nullptr)
	, structure_(std::move(structure))
	{}

	aggnode(const aggnode & other) = delete;

	aggnode(aggnode && other) = delete;

	aggnode &
	operator=(const aggnode & other) = delete;

	aggnode &
	operator=(aggnode && other) = delete;

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
	add_child(std::unique_ptr<aggnode> child)
	{
		children_.emplace_back(std::move(child));
		children_[nchildren()-1]->parent_ = this;
	}

	inline aggnode *
	child(size_t n) const noexcept
	{
		JLM_DEBUG_ASSERT(n < nchildren());
		return children_[n].get();
	}

	inline aggnode *
	parent() noexcept
	{
		return parent_;
	}

	inline const jlm::agg::structure &
	structure() const noexcept
	{
		return *structure_;
	}

	virtual std::string
	debug_string() const = 0;

private:
	aggnode * parent_;
	std::vector<std::unique_ptr<aggnode>> children_;
	std::unique_ptr<jlm::agg::structure> structure_;
};

template <class T> static inline bool
is(const agg::aggnode * node)
{
	static_assert(std::is_base_of<agg::aggnode, T>::value,
		"Template parameter T must be derived from jlm::aggnode");

	return dynamic_cast<const T*>(node) != nullptr;
}

/* entry node class */

class entryaggnode final : public aggnode {
public:
	virtual
	~entryaggnode();

	inline
	entryaggnode(const jlm::entry & attribute)
	: aggnode(std::make_unique<agg::entry>(attribute))
	{}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<agg::aggnode>
	create(const jlm::entry & attribute)
	{
		return std::make_unique<entryaggnode>(attribute);
	}
};

/* exit node class */

class exitaggnode final : public aggnode {
public:
	virtual
	~exitaggnode();

	inline
	exitaggnode(const jlm::exit & attribute)
	: aggnode(std::make_unique<agg::exit>(attribute))
	{}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<agg::aggnode>
	create(const jlm::exit & attribute)
	{
		return std::make_unique<exitaggnode>(attribute);
	}
};

/* basic block node class */

class blockaggnode final : public aggnode {
public:
	virtual
	~blockaggnode();

	inline
	blockaggnode(jlm::basic_block && bb)
	: aggnode(std::make_unique<agg::block>(std::move(bb)))
	{}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<agg::aggnode>
	create(jlm::basic_block && bb)
	{
		return std::make_unique<blockaggnode>(std::move(bb));
	}
};

/* linear node class */

class linearaggnode final : public aggnode {
public:
	virtual
	~linearaggnode();

	inline
	linearaggnode(
		std::unique_ptr<agg::aggnode> n1,
		std::unique_ptr<agg::aggnode> n2)
	: aggnode(std::make_unique<linear>())
	{
		add_child(std::move(n1));
		add_child(std::move(n2));
	}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<agg::aggnode>
	create(
		std::unique_ptr<agg::aggnode> n1,
		std::unique_ptr<agg::aggnode> n2)
	{
		return std::make_unique<linearaggnode>(std::move(n1), std::move(n2));
	}
};

/* branch node class */

class branchaggnode final : public aggnode {
public:
	virtual
	~branchaggnode();

	inline
	branchaggnode(std::unique_ptr<agg::aggnode> split)
	: aggnode(std::make_unique<agg::branch>())
	{
		add_child(std::move(split));
	}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<agg::aggnode>
	create(std::unique_ptr<agg::aggnode> split)
	{
		return std::make_unique<branchaggnode>(std::move(split));
	}
};

/* loop node class */

class loopaggnode final : public aggnode {
public:
	virtual
	~loopaggnode();

	inline
	loopaggnode(std::unique_ptr<agg::aggnode> body)
	: aggnode(std::make_unique<agg::loop>())
	{
		add_child(std::move(body));
	}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<agg::aggnode>
	create(std::unique_ptr<agg::aggnode> body)
	{
		return std::make_unique<loopaggnode>(std::move(body));
	}
};

}}

#endif
