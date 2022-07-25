/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_HASHSET_HPP
#define JLM_UTIL_HASHSET_HPP

#include <unordered_set>

namespace jlm
{

/**
 * Represents a set of values. A set is a collection that contains no duplicate elements, and whose elements are in
 * no particular order.
 * @tparam T The type of the items in the hash set.
 */
template<typename T>
class HashSet
{
  /**
   * Determines whether an HashSet object contains the specified item.
   *
   * @param item The item to locate in the HashSet object.
   * @return True if the HashSet object contains \p item, otherwise false.
   */
  virtual bool
  Contains(T item) const noexcept = 0;

  /**
   * Get the number of items contained in the set.
   *
   * @return The number of items contained in the set.
   */
  [[nodiscard]] virtual std::size_t
  Size() const noexcept = 0;

  /**
   * Inserts the specified item to a set.
   *
   * @param item The item to add.
   * @return True if \p item is added to the HashSet object. False if \p item is already present.
   */
  virtual bool
  Insert(T item) = 0;

  /**
   * Removes all items from a HashSet object.
   */
  virtual void
  Clear() noexcept = 0;

  /**
   * Removes the specified item from a HashSet object.
   *
   * @param item The item to remove.
   * @return True if \p item is successfully found and removed. False if \p item is not found.
   */
  virtual bool
  Remove(T item) = 0;
};

template<typename T>
class StdUnorderedSetHashSet final : HashSet<T>
{
public:
  ~StdUnorderedSetHashSet() noexcept
  = default;

  StdUnorderedSetHashSet()
  = default;

  StdUnorderedSetHashSet(std::initializer_list<T> initializerList)
  : Set_(initializerList)
  {}

  StdUnorderedSetHashSet(const StdUnorderedSetHashSet & other)
  : Set_(other.Set_)
  {}

  StdUnorderedSetHashSet(StdUnorderedSetHashSet && other) noexcept
  : Set_(std::move(other.Set_))
  {}

  StdUnorderedSetHashSet&
  operator=(const StdUnorderedSetHashSet & other)
  {
    Set_ = other.Set_;
    return *this;
  }

  StdUnorderedSetHashSet&
  operator=(StdUnorderedSetHashSet && other) noexcept
  {
    Set_ = std::move(other.Set_);
    return *this;
  }

  bool
  Contains(T item) const noexcept override
  {
    return Set_.find(item) != Set_.end();
  }

  [[nodiscard]] std::size_t
  Size() const noexcept override
  {
    return Set_.size();
  }

  bool
  Insert(T item) override
  {
    auto size = Size();
    Set_.insert(std::move(item));
    return (size != Size());
  }

  void
  Clear() noexcept override
  {
    Set_.clear();
  }

  bool
  Remove(T item) override
  {
    return Set_.erase(item) != 0;
  }

private:
  std::unordered_set<T> Set_;
};

}

#endif //JLM_UTIL_HASHSET_HPP