/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_PTRITERATOR_HPP
#define JLM_UTIL_PTRITERATOR_HPP

#include <vector>

namespace jlm::util
{

/**
 * Helper class for providing iterators over lists of pointers and/or smart pointers.
 * The iterator does one level of pointer dereferencing when yielding elements.
 * This means the user may not modify the pointers, but may modify the elements.
 * To get a const interator, let T be a const type.
 * This iterator should not be used if any element may be a null pointer.
 *
 * @tparam T the underlying type.
 * @tparam BaseIterator the type of the base iterator, can always be a const iterator.
 */
template<typename T, typename BaseIterator>
class PtrIterator final
{
public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::forward_iterator_tag;

  explicit PtrIterator(BaseIterator it)
      : Iterator_(it)
  {}

  PtrIterator &
  operator++() noexcept
  {
    ++Iterator_;
    return *this;
  }

  PtrIterator
  operator++(int) noexcept
  {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const PtrIterator & other) const noexcept
  {
    return Iterator_ == other.Iterator_;
  }

  bool
  operator!=(const PtrIterator & other) const noexcept
  {
    return !(other == *this);
  }

  reference
  operator*() const noexcept
  {
    return *(*Iterator_);
  }

  pointer
  operator->() const noexcept
  {
    return &(*(*Iterator_));
  }

private:
  BaseIterator Iterator_;
};
}

#endif // JLM_UTIL_PTRITERATOR_HPP
