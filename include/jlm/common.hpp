/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_COMMON_HPP
#define JLM_COMMON_HPP

#include <assert.h>

#include <stdexcept>

#define JLM_ASSERT(x) assert(x)

#ifdef JLM_DEBUG
#	define JLM_DEBUG_ASSERT(x) assert(x)
#else
#	define JLM_DEBUG_ASSERT(x) (void)(x)
#endif

namespace jlm {

class error : public std::runtime_error {
public:
	virtual
	~error() noexcept;

	inline
	error(const std::string & msg)
	: std::runtime_error(msg)
	{}
};

}

#endif
