/*
 * Copyright 2013 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_UTIL_DOUBLE_HPP
#define JIVE_UTIL_DOUBLE_HPP

#include <jive/common.hpp>

#include <stdbool.h>
#include <stdint.h>

static inline uint64_t
jive_double_raw_mantissa(double d)
{
	union {
		uint64_t int_value;
		double double_value;
	} u;
	u.double_value = d;
	return (u.int_value & 0xFFFFFFFFFFFFFL);
}

static inline uint16_t
jive_double_raw_exponent(double d)
{
	union {
		uint64_t int_value;
		double double_value;
	} u;
	u.double_value = d;
	return (u.int_value >> 52) & 0x7FF;
}

static inline bool
jive_double_is_signed(double d)
{
	union {
		uint64_t int_value;
		double double_value;
	} u;
	u.double_value = d;
	return (u.int_value >> 63);
}

static inline bool
jive_double_is_normalized(double d)
{
	uint16_t e = jive_double_raw_exponent(d);
	if ( e != 0 && e != 0x7FF)
		return true;

	return false;
}

static inline bool
jive_double_is_zero(double d)
{
	if (jive_double_raw_exponent(d) == 0 && jive_double_raw_mantissa(d) == 0)
		return true;

	return false;
}

static inline bool
jive_double_is_signed_zero(double d)
{
	if (jive_double_is_zero(d) && jive_double_is_signed(d))
		return true;

	return false;
}

static inline bool
jive_double_is_unsigned_zero(double d)
{
	if (jive_double_is_zero(d) && !jive_double_is_signed(d))
		return true;

	return false;
}

static inline bool
jive_double_is_infinity(double d)
{
	if (jive_double_raw_exponent(d) == 0x7FF && jive_double_raw_mantissa(d) == 0)
		return true;

	return false;
}

static inline bool
jive_double_is_signed_infinity(double d)
{
	if (jive_double_is_infinity(d) && jive_double_is_signed(d))
		return true;

	return false;
}

static inline bool
jive_double_is_unsigned_infinity(double d)
{
	if (jive_double_is_infinity(d) && !jive_double_is_signed(d))
		return true;

	return false;
}

static inline bool
jive_double_is_nan(double d)
{
	if (jive_double_raw_exponent(d) == 0x7FF && jive_double_raw_mantissa(d) != 0)
		return true;

	return false;
}

static inline bool
jive_double_is_value(double d)
{
	if (jive_double_raw_exponent(d) == 0x7FF)
		return false;

	return true;
}

static inline uint64_t
jive_double_decoded_mantissa(double d)
{
	JIVE_DEBUG_ASSERT(jive_double_is_value(d));

	uint64_t m = jive_double_raw_mantissa(d);
	if (jive_double_is_normalized(d))
		m |= 0x10000000000000L;

	return m;
}

static inline int16_t
jive_double_decoded_exponent(double d)
{
	JIVE_DEBUG_ASSERT(jive_double_is_value(d));

	if (!jive_double_is_normalized(d))
		return -1022;

	int32_t i = jive_double_raw_exponent(d);
	return i - 1023;
}

#endif
