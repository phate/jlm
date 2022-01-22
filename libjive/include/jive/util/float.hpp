/*
 * Copyright 2013 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_UTIL_FLOAT_HPP
#define JIVE_UTIL_FLOAT_HPP

#include <jive/common.hpp>

#include <stdbool.h>
#include <stdint.h>

static inline uint32_t
jive_float_raw_mantissa(float f)
{
	union {
		uint32_t int_value;
		float float_value;
	} u;
	u.float_value = f;
	return (u.int_value & 0x007FFFFF);
}

static inline uint8_t
jive_float_raw_exponent(float f)
{
	union {
		uint32_t int_value;
		float float_value;
	} u;
	u.float_value = f;
	return ((u.int_value & 0x7F800000) >> 23);
}

static inline bool
jive_float_is_signed(float f)
{
	union {
		uint32_t int_value;
		float float_value;
	} u;
	u.float_value = f;
	return (u.int_value >> 31);
}

static inline bool
jive_float_is_normalized(float f)
{
	uint8_t e = jive_float_raw_exponent(f);
	if (e != 0 && e != 255)
		return true;

	return false;
}

static inline bool
jive_float_is_zero(float f)
{
	if (jive_float_raw_exponent(f) == 0 && jive_float_raw_mantissa(f) == 0)
		return true;

	return false;
}

static inline bool
jive_float_is_signed_zero(float f)
{
	if (jive_float_is_zero(f) && jive_float_is_signed(f))
		return true;

	return false;
}

static inline bool
jive_float_is_unsigned_zero(float f)
{
	if (jive_float_is_zero(f) && !jive_float_is_signed(f))
		return true;

	return false;
}

static inline bool
jive_float_is_infinity(float f)
{
	if (jive_float_raw_exponent(f) == 255 && jive_float_raw_mantissa(f) == 0)
		return true;

	return false;
}

static inline bool
jive_float_is_signed_infinity(float f)
{
	if (jive_float_is_infinity(f) && jive_float_is_signed(f))
		return true;

	return false;
}

static inline bool
jive_float_is_unsigned_infinity(float f)
{
	if (jive_float_is_infinity(f) && !jive_float_is_signed(f))
		return true;

	return false;
}

static inline bool
jive_float_is_nan(float f)
{
	if (jive_float_raw_exponent(f) == 255 && jive_float_raw_mantissa(f) != 0)
		return true;

	return false;
}

static inline bool
jive_float_is_value(float f)
{
	if (jive_float_raw_exponent(f) == 255)
		return false;

	return true;
}

static inline uint32_t
jive_float_decoded_mantissa(float f)
{
	JIVE_DEBUG_ASSERT(jive_float_is_value(f));

	uint32_t m = jive_float_raw_mantissa(f);
	if (jive_float_is_normalized(f))
		m |= 0x00800000;

	return m;
}

static inline int8_t
jive_float_decoded_exponent(float f)
{
	JIVE_DEBUG_ASSERT(jive_float_is_value(f));

	if (!jive_float_is_normalized(f))
		return -126;

	int16_t i = jive_float_raw_exponent(f);
	return i - 127;
}

#endif
