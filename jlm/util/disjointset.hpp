/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_DISJOINTSET_HPP
#define JLM_UTIL_DISJOINTSET_HPP

#include <jlm/util/common.hpp>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace jlm {

template <class T>
class disjointset final {
public:
	class set;

private:
  class member_iterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

  private:
			friend class disjointset::set;

			member_iterator(const set * node)
			: node_(node)
			{}

		public:
			reference
			operator*() const
			{
				JLM_ASSERT(node_ != nullptr);
				return node_->value();
			}

			pointer
			operator->() const
			{
				return &operator*();
			}

			member_iterator &
			operator++()
			{
				JLM_ASSERT(node_ != nullptr);

				node_ = node_->is_root() ? nullptr : node_->next_;
				return *this;
			}

			member_iterator
			operator++(int)
			{
				member_iterator tmp = *this;
				++*this;
				return tmp;
			}

			bool
			operator==(const member_iterator & other) const
			{
				return node_ == other.node_;
			}

			bool
			operator!=(const member_iterator & other) const
			{
				return !operator==(other);
			}

		private:
			const set * node_;
	};

public:
	class set final {
		friend class disjointset;

		private:
			set(const T & value)
			: value_(value)
			, size_(1)
			, next_(this)
			, parent_(this)
			{}

			set(const set&) = delete;

			set(set && other) = delete;

			set &
			operator=(const set&) = delete;

			set &
			operator=(set && other) = delete;

			static std::unique_ptr<set>
			create(const T & value)
			{
				return std::unique_ptr<set>(new set(value));
			}

			const set *
			root() const noexcept
			{
				auto child = this;
				auto parent = parent_;

				/* path halving */
				while (parent != child) {
					parent_ = parent->parent_;
					child = parent;
					parent = parent_;
				}

				return child;
			}

		public:
			bool
			operator==(const set & other) const noexcept
			{
				return value_ == other.value_;
			}

			bool
			operator!=(const set & other) const noexcept
			{
				return !operator==(other);
			}

			member_iterator
			begin() const
			{
				return member_iterator(root()->next_);
			}

			member_iterator
			end() const
			{
				return member_iterator(nullptr);
			}

			size_t
			nmembers() const noexcept
			{
				return root()->size_;
			}

			const T &
			value() const noexcept
			{
				return value_;
			}

			bool
			is_root() const noexcept
			{
				return this == parent_;
			}

		private:
			T value_;
			mutable size_t size_;
			mutable const set * next_;
			mutable const set * parent_;
	};

  class set_iterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const set*;
    using difference_type = ptrdiff_t;
    using pointer = const set**;
    using reference = const set*&;

  private:
			friend class disjointset;

			set_iterator(const typename std::unordered_set<const set*>::const_iterator & it)
			: it_(it)
			{}

		public:
			const set &
			operator*() const
			{
				JLM_ASSERT(*it_ != nullptr);
				return **it_;
			}

			const set *
			operator->() const
			{
				return &operator*();
			}

			set_iterator &
			operator++()
			{
				++it_;
				return *this;
			}

			set_iterator
			operator++(int)
			{
				member_iterator tmp = *this;
				++*this;
				return tmp;
			}

			bool
			operator==(const set_iterator & other) const
			{
				return it_ == other.it_;
			}

			bool
			operator!=(const set_iterator & other) const
			{
				return !operator==(other);
			}

		private:
			typename std::unordered_set<const set*>::const_iterator it_;
	};

public:
	constexpr
	disjointset()
	{}

	disjointset(const std::vector<T> & elements)
	{
		insert(elements);
	}

	disjointset(const disjointset & other)
	{
		operator=(other);
	}

	disjointset(disjointset && other)
	{
		operator=(other);
	}

	disjointset &
	operator=(const disjointset & other)
	{
		if (this == &other)
			return *this;

		roots_ = other.roots_;
		values_ = other.values_;

		return *this;
	}

	disjointset &
	operator=(disjointset && other)
	{
		if (this == &other)
			return *this;

		roots_ = std::move(other.roots_);
		values_= std::move(other.values_);

		return *this;
	}

	void
	insert(const std::vector<T> & elements)
	{
		for (auto & element : elements)
			insert(element);
	}

	const set *
	insert(const T & element)
	{
		if (contains(element))
			return values_.find(element)->second.get();

		values_[element] = set::create(element);
		auto s = values_.find(element)->second.get();
		roots_.insert(s);
		return s;
	}

	set_iterator
	begin() const
	{
		return roots_.begin();
	}

	set_iterator
	end() const
	{
		return roots_.end();
	}

	bool
	empty() const noexcept
	{
		return values_.empty();
	}

	size_t
	nvalues() const noexcept
	{
		return values_.size();
	}

	size_t
	nsets() const noexcept
	{
		return roots_.size();
	}

	void
	clear()
	{
		roots_.clear();
		values_.clear();
	}

	/*
		@brief Find the representative set for the set containing \p element.
	*/
	const set *
	find(const T & element) const noexcept
	{
		JLM_ASSERT(contains(element));

		return values_.find(element)->second->root();
	}

	/*
		@brief Find the representative set for the set containing \p element
		       if \p element is present. Otherwise, insert \p element and
		       return the representative set.
	*/
	const set *
	find_or_insert(const T & element) noexcept
	{
		if (!contains(element))
			return insert(element);

		return find(element);
	}

	/*
		@brief Union the two sets containing elements \p e1 and \p e2.
	*/
	const set *
	merge(const T & e1, const T & e2)
	{
		auto root1 = find(e1);
		auto root2 = find(e2);

		/* Both elements are already in the same set. */
		if (root1 == root2)
			return root1;

		/* union by size */
		auto size1 = root1->size_;
		auto size2 = root2->size_;
		if (size1 < size2)
			std::swap(root1, root2);

		/* Insert elements of smaller set into bigger set */
		std::swap(root1->next_, root2->next_);

		roots_.erase(root2);
		root2->parent_ = root1;
		root1->size_ = size1 + size2;

		return root1;
	}

private:
	bool
	contains(const T & element) const
	{
		return values_.find(element) != values_.end();
	}

	std::unordered_set<const set*> roots_;
	std::unordered_map<T, std::unique_ptr<set>> values_;
};

}

#endif
