/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_ITERATOR_RANGE_HPP
#define JLM_UTIL_ITERATOR_RANGE_HPP

#include <utility>

namespace jlm {

/** \brief Iterator Range
*
* A range-compatible interface wrapping a pair of iterators.
*/
template<typename T>
class iterator_range {
public:
	iterator_range(T begin, T end)
	: begin_(std::move(begin))
	, end_(std::move(end))
	{}

	template<typename container_t>
	iterator_range(container_t && c)
	: begin_(c.begin())
	, end_(c.end())
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

private:
	T begin_, end_;
};

}

#endif
