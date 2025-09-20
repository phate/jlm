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

namespace jlm::util
{

template<class T>
class DisjointSet final
{
public:
  class Set;

private:
  class MemberIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T *;
    using reference = const T &;

  private:
    friend class DisjointSet::Set;

    explicit MemberIterator(const Set * node)
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

    MemberIterator &
    operator++()
    {
      JLM_ASSERT(node_ != nullptr);

      node_ = node_->is_root() ? nullptr : node_->next_;
      return *this;
    }

    MemberIterator
    operator++(int)
    {
      MemberIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const MemberIterator & other) const
    {
      return node_ == other.node_;
    }

    bool
    operator!=(const MemberIterator & other) const
    {
      return !operator==(other);
    }

  private:
    const Set * node_;
  };

public:
  class Set final
  {
    friend class DisjointSet;

  private:
    Set(const T & value)
        : value_(value),
          size_(1),
          next_(this),
          parent_(this)
    {}

    Set(const Set &) = delete;

    Set(Set && other) = delete;

    Set &
    operator=(const Set &) = delete;

    Set &
    operator=(Set && other) = delete;

    static std::unique_ptr<Set>
    create(const T & value)
    {
      return std::unique_ptr<Set>(new Set(value));
    }

    const Set *
    root() const noexcept
    {
      auto child = this;
      auto parent = parent_;

      /* path halving */
      while (parent != child)
      {
        parent_ = parent->parent_;
        child = parent;
        parent = parent_;
      }

      return child;
    }

  public:
    bool
    operator==(const Set & other) const noexcept
    {
      return value_ == other.value_;
    }

    bool
    operator!=(const Set & other) const noexcept
    {
      return !operator==(other);
    }

    MemberIterator
    begin() const
    {
      return MemberIterator(root()->next_);
    }

    MemberIterator
    end() const
    {
      return MemberIterator(nullptr);
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
    mutable const Set * next_;
    mutable const Set * parent_;
  };

  class set_iterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Set *;
    using difference_type = ptrdiff_t;
    using pointer = const Set **;
    using reference = const Set *&;

  private:
    friend class DisjointSet;

    set_iterator(const typename std::unordered_set<const Set *>::const_iterator & it)
        : it_(it)
    {}

  public:
    const Set &
    operator*() const
    {
      JLM_ASSERT(*it_ != nullptr);
      return **it_;
    }

    const Set *
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
      MemberIterator tmp = *this;
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
    typename std::unordered_set<const Set *>::const_iterator it_;
  };

public:
  constexpr DisjointSet() = default;

  explicit DisjointSet(const std::vector<T> & elements)
  {
    insert(elements);
  }

  DisjointSet(const DisjointSet & other)
  {
    operator=(other);
  }

  DisjointSet(DisjointSet && other)
  {
    operator=(other);
  }

  DisjointSet &
  operator=(const DisjointSet & other)
  {
    if (this == &other)
      return *this;

    roots_ = other.roots_;
    values_ = other.values_;

    return *this;
  }

  DisjointSet &
  operator=(DisjointSet && other)
  {
    if (this == &other)
      return *this;

    roots_ = std::move(other.roots_);
    values_ = std::move(other.values_);

    return *this;
  }

  void
  insert(const std::vector<T> & elements)
  {
    for (auto & element : elements)
      insert(element);
  }

  const Set *
  insert(const T & element)
  {
    if (contains(element))
      return values_.find(element)->second.get();

    values_[element] = Set::create(element);
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
  const Set *
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
  const Set *
  find_or_insert(const T & element) noexcept
  {
    if (!contains(element))
      return insert(element);

    return find(element);
  }

  /*
    @brief Union the two sets containing elements \p e1 and \p e2.
  */
  const Set *
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

  std::unordered_set<const Set *> roots_;
  std::unordered_map<T, std::unique_ptr<Set>> values_;
};

}

#endif
