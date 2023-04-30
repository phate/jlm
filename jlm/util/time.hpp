/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_TIME_HPP
#define JLM_UTIL_TIME_HPP

#include <chrono>

namespace jlm {

class timer final {
public:
	constexpr timer()
	{}

	void
	start() noexcept
	{
		start_ = std::chrono::high_resolution_clock::now();
	}

	void
	stop() noexcept
	{
		end_ = std::chrono::high_resolution_clock::now();
	}

	size_t
	ns() const
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(end_-start_).count();
	}

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_;
	std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

}

#endif
