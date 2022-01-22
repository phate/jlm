/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */
#ifndef JIVE_UTIL_CSTR_MAP_HPP
#define JIVE_UTIL_CSTR_MAP_HPP

/*
 * Auxiliary internal class template cstr_map<T>, to provide a mapping
 * from c-string (const char *) instances  to other objects.
 */

#include <string.hpp>

#include <unordered_map>

namespace jive {
namespace detail {

struct cstr_eq {
	bool operator()(const char * s1, const char * s2) const noexcept
	{
		return strcmp(s1, s2) == 0;
	}
};

struct cstr_hash {
	size_t operator()(const char * s) const noexcept
	{
		size_t tmp = 0;
		while (*s) {
			tmp = tmp * 5 + *(unsigned char *) s;
			++s;
		}
		return tmp;
	}
};

template<typename T>
using cstr_map = std::unordered_map<const char *, T, cstr_hash, cstr_eq>;

}
}

#endif
