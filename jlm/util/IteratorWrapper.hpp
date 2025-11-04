/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_ITERATORWRAPPER_HPP
#define JLM_UTIL_ITERATORWRAPPER_HPP

#include <jlm/util/common.hpp>

#include <iterator>

namespace jlm::util
{

/**
 * Functor for dereferencing iterators to (smart)pointers.
 * The iterator may for example be a vector<unique_ptr<T>>:: iterator.
 *
 * @tparam T the result type
 * @tparam BaseIterator the iterator type
 */
template<typename T, typename BaseIterator>
struct PtrDereferenceFunc
{
  [[nodiscard]] T &
  operator()(const BaseIterator & it) const
  {
    JLM_ASSERT(*it != nullptr);
    return *(*it);
  }
};

/**
 * Functor for dereferencing iterators to maps, where the values are (smart)pointers.
 * The iterator may for example be an unordered_map<int, unique_ptr<T>>:: iterator.
 *
 * @tparam T the result type
 * @tparam BaseIterator the iterator type
 */
template<typename T, typename BaseIterator>
struct MapValuePtrDereferenceFunc
{
  [[nodiscard]] T &
  operator()(const BaseIterator & it) const
  {
    JLM_ASSERT(it->second != nullptr);
    return *(it->second);
  }
};

/**
 * Functor for iterators to maps, yielding the value
 * The iterator may for example be an unordered_map<int, int>:: iterator.
 *
 * @tparam T the result type
 * @tparam BaseIterator the iterator type
 */
template<typename T, typename BaseIterator>
struct MapValueDereferenceFunc
{
  [[nodiscard]] T &
  operator()(const BaseIterator & it) const
  {
    return it->second;
  }
};

/**
 * Helper class for providing iterators over lists of wrapper types, like (smart)pointers and maps.
 * To get a const interator, let T be a const type.
 * This iterator should not be used if any element in the collection may be a null pointer.
 *
 * @tparam T the underlying type.
 * @tparam BaseIterator the type of the base iterator, can always be a const iterator.
 * @tparam DereferenceFunc a Functor for converting a BaseIterator to a T&
 */
template<typename T, typename BaseIterator, typename DereferenceFunc>
class IteratorWrapper final
{
public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::forward_iterator_tag;

  explicit IteratorWrapper(BaseIterator it)
      : Iterator_(it)
  {}

  IteratorWrapper &
  operator++() noexcept
  {
    ++Iterator_;
    return *this;
  }

  IteratorWrapper
  operator++(int) noexcept
  {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  [[nodiscard]] bool
  operator==(const IteratorWrapper & other) const noexcept
  {
    return Iterator_ == other.Iterator_;
  }

  [[nodiscard]] bool
  operator!=(const IteratorWrapper & other) const noexcept
  {
    return !(other == *this);
  }

  [[nodiscard]] reference
  operator*() const noexcept
  {
    return DereferenceFunc{}(Iterator_);
  }

  [[nodiscard]] pointer
  operator->() const noexcept
  {
    return &operator*();
  }

private:
  BaseIterator Iterator_;
};

/**
 * Wrapper for iterators over (smart)pointers, automatically dereferencing each element
 * @tparam T the type of the elements
 * @tparam BaseIterator the type of the iterator being wrapped
 */
template<typename T, typename BaseIterator>
using PtrIterator = IteratorWrapper<T, BaseIterator, PtrDereferenceFunc<T, BaseIterator>>;

/**
 * Wrapper for iterators over maps where the values are (smart)pointers. Dereferences these values.
 * @tparam T the type of the map's dereferenced value elements
 * @tparam BaseIterator the type of the iterator being wrapped
 */
template<typename T, typename BaseIterator>
using MapValuePtrIterator =
    IteratorWrapper<T, BaseIterator, MapValuePtrDereferenceFunc<T, BaseIterator>>;

/**
 * Wrapper for iterators over values of maps.
 * @tparam T the type of the map's value elements
 * @tparam BaseIterator the type of the iterator being wrapped
 */
template<typename T, typename BaseIterator>
using MapValueIterator =
    IteratorWrapper<T, BaseIterator, MapValueDereferenceFunc<T, BaseIterator>>;

}

#endif // JLM_UTIL_ITERATORWRAPPER_HPP
