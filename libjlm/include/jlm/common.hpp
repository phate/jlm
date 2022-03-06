/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_COMMON_HPP
#define JLM_COMMON_HPP

#include <assert.h>

#include <iostream>
#include <stdexcept>

#ifdef JLM_ENABLE_ASSERTS
#	define JLM_ASSERT(x) assert(x)
#else
#	define JLM_ASSERT(x) do{}while(0 && (x))
#endif

#define JLM_NORETURN __attribute__((noreturn))

namespace jlm {

JLM_NORETURN static inline void
unreachable(const char * msg, const char * file, unsigned line)
{
	if (msg)
		std::cerr << msg << "\n";

	std::cerr << "UNREACHABLE executed";

	if (file)
		std::cerr << " at " << file << ":" << line << "\n";

	abort();
}

}


#define JLM_UNREACHABLE(msg) jlm::unreachable(msg, __FILE__, __LINE__)

namespace jlm {

template <class To, class From> static inline To *
AssertedCast(From * value)
{
  JLM_ASSERT(dynamic_cast<To*>(value));
  return static_cast<To*>(value);
}

class error : public std::runtime_error {
public:
	virtual
	~error();

	inline
	error(const std::string & msg)
	: std::runtime_error(msg)
	{}
};

}

#endif
