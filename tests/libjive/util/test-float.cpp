/*
 * Copyright 2013 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/util/float.hpp>

#include <assert.h>

static inline float
create_float(bool s, uint32_t m, uint8_t e)
{
	uint32_t i = 0;
	if (s) {
		i = (1 << 31) | (((uint32_t)e) << 23) | (m & 0x007FFFFF);
	} else {
		i = (((uint32_t)e) << 23) | (m & 0x007FFFFF);
	}

	return *((float *)&i);
}

static int
test_main(void)
{
	float zero = create_float(false, 0, 0);
	assert(jive_float_is_value(zero));
	assert(jive_float_is_zero(zero));
	assert(jive_float_is_unsigned_zero(zero));
	assert(!jive_float_is_signed_zero(zero));
	assert(!jive_float_is_normalized(zero));

	float mzero = create_float(true, 0, 0);
	assert(jive_float_is_value(mzero));
	assert(jive_float_is_zero(mzero));
	assert(!jive_float_is_unsigned_zero(mzero));
	assert(jive_float_is_signed_zero(mzero));
	assert(!jive_float_is_normalized(mzero));

	float nan = create_float(false, 1, 255);
	assert(!jive_float_is_value(nan));
	assert(jive_float_is_nan(nan));
	assert(!jive_float_is_normalized(nan));

	float inf	= create_float(false, 0, 255);
	assert(!jive_float_is_value(inf));
	assert(jive_float_is_infinity(inf));
	assert(jive_float_is_unsigned_infinity(inf));
	assert(!jive_float_is_signed_infinity(inf));
	assert(!jive_float_is_normalized(inf));

	float minf = create_float(true, 0, 255);
	assert(!jive_float_is_value(minf));
	assert(jive_float_is_infinity(minf));
	assert(!jive_float_is_unsigned_infinity(minf));
	assert(jive_float_is_signed_infinity(minf));
	assert(!jive_float_is_normalized(minf));

	assert(jive_float_is_value(25.0));
	assert(!jive_float_is_infinity(25.0));
	assert(!jive_float_is_zero(25.0));
	assert(!jive_float_is_nan(25.0));
	assert(jive_float_is_normalized(25.0));
	assert(jive_float_decoded_mantissa(25.0) == 0xC80000);
	assert(jive_float_decoded_exponent(25.0) == 4);

	float denorm = 1.40129846e-45;
	assert(jive_float_is_value(denorm));
	assert(!jive_float_is_infinity(denorm));
	assert(!jive_float_is_zero(denorm));
	assert(!jive_float_is_nan(denorm));
	assert(!jive_float_is_normalized(denorm));
	assert(jive_float_decoded_mantissa(denorm) == 1);
	assert(jive_float_decoded_exponent(denorm) == -126);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjive/util/test-float", test_main)
