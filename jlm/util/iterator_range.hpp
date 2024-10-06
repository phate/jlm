/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_ITERATOR_RANGE_HPP
#define JLM_UTIL_ITERATOR_RANGE_HPP

#include <utility>

namespace jlm::util
{

/** \brief Iterator Range
 *
 * A range-compatible interface wrapping a pair of iterators.
 */
template<typename T>
class iterator_range
{
public:
  iterator_range(T begin, T end)
      : begin_(std::move(begin)),
        end_(std::move(end))
  {}

  template<typename container_t>
  iterator_range(container_t && c)
      : begin_(c.begin()),
        end_(c.end())
  {}

  T
  begin() const
  {
    return begin_;
  }

  T
  end() const
  {
    return end_;
  }

  /**
   * Count all elements in this range that match the specified condition \p match.
   *
   * @tparam F A type that supports the function call operator bool operator(const E e), where E
   * is the element type returned by the dereference operator of the iterators in this range.
   * @param match Defines the condition for the elements to be counted.
   * @return The number of times an element in the range fulfilled the condition \p match.
   */
  template<typename F>
  std::size_t
  CountWhere(const F & match)
  {
    std::size_t count = 0;
    for (auto it = begin(); it != end(); it++)
    {
      if (match(*it))
      {
        count++;
      }
    }

    return count;
  }

private:
  T begin_, end_;
};

}

#endif
