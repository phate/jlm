/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_UTIL_BUFFER_HPP
#define JIVE_UTIL_BUFFER_HPP

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <jive/common.hpp>

#include <stdint.h>
#include <vector>

/** \file jive/buffer.h */

namespace jive {

class buffer final {
public:
	inline
	~buffer()
	{}

	inline
	buffer()
	{}

	buffer(const buffer &) = delete;

	buffer(buffer &&) = delete;

	buffer &
	operator=(const buffer &) = delete;

	buffer &
	operator=(buffer &&) = delete;

	inline void
	push_back(const void * data, size_t nbytes)
	{
		auto d = static_cast<const uint8_t*>(data);
		for (size_t n = 0; n < nbytes; n++)
			data_.push_back(d[n]);
	}

	inline void
	push_back(const std::string & s)
	{
		push_back(s.c_str(), s.size());
	}

	inline void
	push_back(uint8_t byte)
	{
		data_.push_back(byte);
	}

	inline size_t
	size() const noexcept
	{
		return data_.size();
	}

	inline const uint8_t *
	data()
	{
		return &data_[0];
	}

	inline const char *
	c_str()
	{
		push_back('\0');
		return (const char *)(&data_[0]);
	}

	inline void
	clear()
	{
		data_.clear();
	}

private:
	std::vector<uint8_t> data_;
};

}

#endif
