/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_COMMON_HPP
#define JIVE_COMMON_HPP

#include <assert.h>

#include <stdexcept>
#include <string>

#define JIVE_ASSERT(x) assert(x)

#ifdef JIVE_DEBUG
#	define JIVE_DEBUG_ASSERT(x) assert(x)
#else
#	define JIVE_DEBUG_ASSERT(x) (void)(x)
#endif

namespace jive {

class compiler_error : public std::runtime_error {
public:
	virtual ~compiler_error() noexcept;

	inline compiler_error(const std::string & arg)
		: std::runtime_error(arg)
	{
	}
};

class type_error : public compiler_error {
public:
	virtual ~type_error() noexcept;

	inline type_error(
		const std::string & expected_type,
		const std::string & received_type)
		: compiler_error(
			"Type error - expected : " + expected_type +
			", received : " + received_type)
	{
	}
};

}

#endif
