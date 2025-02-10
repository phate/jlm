/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_HASHSET_HPP
#define JLM_UTIL_HASHSET_HPP

#include <jlm/util/Hash.hpp>
#include <jlm/util/iterator_range.hpp>

#include <unordered_set>

namespace jlm::util
{

/**
 * Represents a set of values. A set is a collection that contains no duplicate elements, and
 * whose elements are in no particular order.
 * @tparam ItemType The type of the items in the hash set.
 */
template<typename ItemType, typename HashFunctor = Hash<ItemType>>
class HashSet
{
  using InternalSet = std::unordered_set<ItemType, HashFunctor>;

public:
  class ItemConstIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ItemType;
    using difference_type = std::ptrdiff_t;
    using pointer = ItemType *;
    using reference = ItemType &;

  private:
    friend HashSet;

    explicit ItemConstIterator(const typename InternalSet::const_iterator & it)
        : It_(it)
    {}

  public:
    [[nodiscard]] ItemType *
    Item() const noexcept
    {
      return It_.operator->();
    }

    const ItemType &
    operator*() const
    {
      return It_.operator*();
    }

    ItemType *
    operator->() const
    {
      return Item();
    }

    ItemConstIterator &
    operator++()
    {
      ++It_;
      return *this;
    }

    ItemConstIterator
    operator++(int)
    {
      ItemConstIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const ItemConstIterator & other) const
    {
      return It_ == other.It_;
    }

    bool
    operator!=(const ItemConstIterator & other) const
    {
      return !operator==(other);
    }

  private:
    typename InternalSet::const_iterator It_;
  };

  ~HashSet() noexcept = default;

  HashSet() = default;

  template<class InputIt>
  HashSet(InputIt begin, InputIt end)
      : Set_(begin, end)
  {}

  HashSet(const HashSet & other)
      : Set_(other.Set_)
  {}

  HashSet(HashSet && other) noexcept
      : Set_(std::move(other.Set_))
  {}

  HashSet(std::initializer_list<ItemType> initializerList)
      : Set_(initializerList)
  {}

  template<typename OtherHashFunctor>
  explicit HashSet(const std::unordered_set<ItemType, OtherHashFunctor> & other)
      : Set_(other.begin(), other.end())
  {}

  HashSet &
  operator=(const HashSet & other)
  {
    Set_ = other.Set_;
    return *this;
  }

  HashSet &
  operator=(HashSet && other) noexcept
  {
    Set_ = std::move(other.Set_);
    return *this;
  }

  /**
   * Removes all items from a HashSet object.
   */
  void
  Clear() noexcept
  {
    Set_.clear();
  }

  /**
   * Determines whether an HashSet object contains the specified item.
   *
   * @param item The item to locate in the HashSet object.
   * @return True if the HashSet object contains \p item, otherwise false.
   */
  bool
  Contains(const ItemType & item) const noexcept
  {
    return Set_.find(item) != Set_.end();
  }

  /**
   * Determines whether a HashSet object is a subset of \p other.
   *
   * @param other The HashSet to compare to the collection.
   * @return Returns true if the collection is a subset of \p other or equal to \p other, otherwise
   * false.
   */
  bool
  IsSubsetOf(const HashSet<ItemType> & other) const noexcept
  {
    if (Size() > other.Size())
    {
      return false;
    }

    for (auto & item : other.Items())
    {
      if (!other.Contains(item))
      {
        return false;
      }
    }

    return true;
  }

  /**
   * Get the number of items contained in the set.
   *
   * @return The number of items contained in the set.
   */
  [[nodiscard]] std::size_t
  Size() const noexcept
  {
    return Set_.size();
  }

  /**
   * Determines whether the set is empty.
   *
   * @return True if the set is empty, otherwise false.
   */
  [[nodiscard]] bool
  IsEmpty() const noexcept
  {
    return Size() == 0;
  }

  /**
   * Inserts the specified item to a set.
   *
   * @param item The item to add.
   * @return True if \p item is added to the HashSet object. False if \p item is already present.
   */
  bool
  Insert(ItemType item)
  {
    auto size = Size();
    Set_.emplace(std::move(item));
    return (size != Size());
  }

  /**
   * Get a iterator_range for iterating through the items in the HashSet.
   *
   * @return A iterator_range.
   */
  [[nodiscard]] IteratorRange<ItemConstIterator>
  Items() const noexcept
  {
    return { ItemConstIterator(Set_.begin()), ItemConstIterator(Set_.end()) };
  }

  /**
   * Modifies this HashSet object to contain all elements that are present in itself, \p other, or
   * both.
   *
   * @param other A HashSet to union with.
   * @return true, if elements were added to this HashSet, otherwise false
   */
  bool
  UnionWith(const HashSet<ItemType> & other)
  {
    const size_t sizeBefore = Size();
    for (auto & item : other.Items())
      Insert(item);
    return sizeBefore != Size();
  }

  /**
   * Modifies this HashSet object to contain all elements in either itself, \p other, or both.
   * Consumes \p other, making it empty.
   *
   * @param other the HashSet to be consumed
   * @return true if elements were added to this HashSet, otherwise false
   */
  bool
  UnionWithAndClear(HashSet<ItemType> & other)
  {
    // Make *this the largest of the two sets, to make the union cheaper
    if (Size() < other.Size())
      std::swap(*this, other);

    bool result = UnionWith(other);
    other.Clear();
    return result;
  }

  /**
   * Modifies this HashSet object to contain only elements that are present in itself and \p other.
   *
   * @param other A HashSet to intersect with.
   */
  void
  IntersectWith(const HashSet<ItemType> & other)
  {
    auto isContained = [&](const ItemType item)
    {
      return !other.Contains(item);
    };

    RemoveWhere(isContained);
  }

  /**
   * Modifies this HashSet object to contain only elements both in itself and \p other.
   * Consumes \p other, making it empty.
   *
   * @param other the HashSet to be consumed
   */
  void
  IntersectWithAndClear(HashSet<ItemType> & other)
  {
    // Make *this the smallest of the two sets, to make the intersection cheaper
    if (Size() > other.Size())
      std::swap(*this, other);

    IntersectWith(other);
    other.Clear();
  }

  /**
   * Modifies this HashSet object by removing any elements that are present in \p other.
   *
   * @param other the HashSet used as the negative side of the set difference.
   */
  void
  DifferenceWith(const HashSet<ItemType> & other)
  {
    // If this HashSet is smaller, loop over it and remove elements in other.
    // If other is smaller, loop over it and remove elements from this.

    if (Size() <= other.Size())
    {
      // This branch also handles the unlikely case where this and other are the same set.

      auto inOther = [&](const ItemType item)
      {
        return other.Contains(item);
      };

      RemoveWhere(inOther);
    }
    else
    {
      for (auto & item : other.Set_)
        Remove(item);
    }
  }

  /**
   * Removes the specified item from a HashSet object.
   *
   * @param item The item to remove.
   * @return True if \p item is successfully found and removed. False if \p item is not found.
   */
  bool
  Remove(ItemType item)
  {
    return Set_.erase(item) != 0;
  }

  /**
   * Removes all elements that match the conditions defined by the specified \p match from a HashSet
   * object.
   *
   * @tparam F A type supporting function call operator: bool operator(const ItemType&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of elements that were removed from the HashSet object.
   */
  template<typename F>
  size_t
  RemoveWhere(const F & match)
  {
    size_t numRemoved = 0;
    auto it = Set_.begin();
    while (it != Set_.end())
    {
      if (match(*it))
      {
        it = Set_.erase(it);
        numRemoved++;
      }
      else
      {
        it++;
      }
    }

    return numRemoved;
  }

  /**
   * Removes the element pointed to by the given iterator
   * @param iterator the element to remove
   * @return an iterator to the element after the removed element
   */
  ItemConstIterator
  Erase(ItemConstIterator iterator)
  {
    return ItemConstIterator(Set_.erase(iterator.It_));
  }

  /**
   * Compares the items of this HashSet object with the items of \p other for equality.
   *
   * @param other HashSet object the items are compared with.
   * @return True, if the items of \p other equal the items of this HashSet object.
   */
  bool
  operator==(const HashSet<ItemType> & other) const noexcept
  {
    if (Size() != other.Size())
      return false;

    for (auto & item : Set_)
      if (!other.Contains(item))
        return false;

    return true;
  }

  /**
   * Compares the items of this HashSet object with the items of \p other for inequality.
   *
   * @param other HashSet object the items are compared with.
   * @return True, if the items of \p other are unequal the items of this HashSet object.
   */
  bool
  operator!=(const HashSet<ItemType> & other) const noexcept
  {
    return !operator==(other);
  }

private:
  InternalSet Set_;
};

}

#endif // JLM_UTIL_HASHSET_HPP
