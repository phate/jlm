/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_AGGREGATION_HPP
#define JLM_LLVM_IR_AGGREGATION_HPP

#include <jlm/llvm/ir/basic-block.hpp>

#include <memory>

namespace jlm {

class cfg;

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

	aggnode()
	: index_(0)
	, parent_(nullptr)
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
		size_t index = nchildren();
		children_.emplace_back(std::move(child));
		children_[index]->parent_ = this;
		children_[index]->index_ = index;
	}

	inline aggnode *
	child(size_t n) const noexcept
	{
		JLM_ASSERT(n < nchildren());
		return children_[n].get();
	}

	aggnode *
	parent() noexcept
	{
		JLM_ASSERT(parent_->child(index_) == this);
		return parent_;
	}

	const aggnode *
	parent() const noexcept
	{
		JLM_ASSERT(parent_->child(index_) == this);
		return parent_;
	}

	size_t
	index() const noexcept
	{
		JLM_ASSERT(parent()->child(index_) == this);
		return index_;
	}

	/**
	* Return the number of nodes of the entire subtree.
	*/
	size_t
	nnodes() const noexcept
	{
		size_t n = 1;
		for (auto & child : children_)
			n += child->nnodes();

		return n;
	}

	virtual std::string
	debug_string() const = 0;

	/** Normalizes an aggregation tree

	This function normalizes an aggregation tree by reducing nested linear nodes to a single linear
	node. For example, the tree:

	linear
	- linear
	-- block
	-- block
	- block

	is reduced to:

	linear
	- block
	- block
	- block

	*/
	static void
	normalize(aggnode & node);

private:
	void
	remove_children()
	{
		children_.clear();
	}

	size_t index_;
	aggnode * parent_;
	std::vector<std::unique_ptr<aggnode>> children_;
};

template <class T> static inline bool
is(const aggnode * node)
{
	static_assert(std::is_base_of<aggnode, T>::value,
		"Template parameter T must be derived from jlm::aggnode");

	return dynamic_cast<const T*>(node) != nullptr;
}

/* entry node class */

class entryaggnode final : public aggnode {
	class constiterator;

public:
	virtual
	~entryaggnode();

	entryaggnode(const std::vector<jlm::argument*> & arguments)
	: arguments_(arguments)
	{}

	constiterator
	begin() const;

	constiterator
	end() const;

	const jlm::argument *
	argument(size_t index) const noexcept
	{
		JLM_ASSERT(index < narguments());
		return arguments_[index];
	}

	size_t
	narguments() const noexcept
	{
		return arguments_.size();
	}

	virtual std::string
	debug_string() const override;

	static std::unique_ptr<aggnode>
	create(const std::vector<jlm::argument*> & arguments)
	{
		return std::make_unique<entryaggnode>(arguments);
	}

private:
	std::vector<jlm::argument*> arguments_;
};

class entryaggnode::constiterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = const jlm::argument*;
  using difference_type = std::ptrdiff_t;
  using pointer = const jlm::argument**;
  using reference = const jlm::argument*&;

	constexpr
	constiterator(const std::vector<jlm::argument*>::const_iterator & it)
	: it_(it)
	{}

	const jlm::argument &
	operator*() const
	{
		return *operator->();
	}

	const jlm::argument *
	operator->() const
	{
		return *it_;
	}

	constiterator &
	operator++()
	{
		it_++;
		return *this;
	}

	constiterator
	operator++(int)
	{
		constiterator tmp = *this;
		it_++;
		return tmp;
	}

	bool
	operator==(const constiterator & other) const
	{
		return it_ == other.it_;
	}

	bool
	operator!=(const constiterator & other) const
	{
		return !operator==(other);
	}

private:
	std::vector<jlm::argument*>::const_iterator it_;
};

/* exit node class */

class exitaggnode final : public aggnode {
	typedef std::vector<const variable*>::const_iterator const_iterator;
public:
	virtual
	~exitaggnode();

	inline
	exitaggnode(const std::vector<const variable*> & results)
	: results_(results)
	{}

	const_iterator
	begin() const
	{
		return results_.begin();
	}

	const_iterator
	end() const
	{
		return results_.end();
	}

	const variable *
	result(size_t index) const noexcept
	{
		JLM_ASSERT(index < nresults());
		return results_[index];
	}

	size_t
	nresults() const noexcept
	{
		return results_.size();
	}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<aggnode>
	create(const std::vector<const variable*> & results)
	{
		return std::make_unique<exitaggnode>(results);
	}

private:
	std::vector<const variable*> results_;
};

/* basic block node class */

class blockaggnode final : public aggnode {
public:
	virtual
	~blockaggnode();

	blockaggnode()
	{}

	inline
	blockaggnode(jlm::taclist && bb)
	: bb_(std::move(bb))
	{}

	virtual std::string
	debug_string() const override;

	inline const jlm::taclist &
	tacs() const noexcept
	{
		return bb_;
	}

	static std::unique_ptr<aggnode>
	create()
	{
		return std::make_unique<blockaggnode>();
	}

	static inline std::unique_ptr<aggnode>
	create(jlm::taclist && bb)
	{
		return std::make_unique<blockaggnode>(std::move(bb));
	}

private:
	jlm::taclist bb_;
};

/* linear node class */

class linearaggnode final : public aggnode {
public:
	virtual
	~linearaggnode();

	inline
	linearaggnode(
		std::unique_ptr<aggnode> n1,
		std::unique_ptr<aggnode> n2)
	{
		add_child(std::move(n1));
		add_child(std::move(n2));
	}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<aggnode>
	create(
		std::unique_ptr<aggnode> n1,
		std::unique_ptr<aggnode> n2)
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
	branchaggnode()
	{}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<aggnode>
	create()
	{
		return std::make_unique<branchaggnode>();
	}
};

/* loop node class */

class loopaggnode final : public aggnode {
public:
	virtual
	~loopaggnode();

	inline
	loopaggnode(std::unique_ptr<aggnode> body)
	{
		add_child(std::move(body));
	}

	virtual std::string
	debug_string() const override;

	static inline std::unique_ptr<aggnode>
	create(std::unique_ptr<aggnode> body)
	{
		return std::make_unique<loopaggnode>(std::move(body));
	}
};

/** \brief Aggregate a properly structured CFG to a aggregation tree.
*
* This function reduces a properly structured CFG to an aggregation tree. The CFG is only
* allowed to consist of the following subgraphs:
*
* 1. Linear subgraphs, such as:
* \dot
* 	digraph linear {
* 		A -> B;
* 	}
* \enddot
*
* 2. Branch subgraphs, such as:
* \dot
* 	\digraph branches {
* 		Split -> A;
* 		Split -> B;
* 		Split -> C;
* 		A -> Join;
* 		B -> Join;
* 		C -> Join;
* 	}
* \enddot
*
* 3. Tail-Controlled Loops, such as:
* \dot
* 	digraph tcloop {
* 		A -> Loop;
* 		Loop -> Loop;
* 		Loop -> B;
* 	}
* \enddot
*
* These subgraphs can be arbitrarily nested. Please refer to Reissmann et al. - RVSDG: An
* Intermediate Representation for Optimizing Compilers [https://doi.org/10.1145/3391902] for more
* information.
*/
std::unique_ptr<aggnode>
aggregate(jlm::cfg & cfg);

size_t
ntacs(const jlm::aggnode & root);

}

#endif
