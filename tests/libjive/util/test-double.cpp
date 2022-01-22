/*
 * Copyright 2013 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/util/double.hpp>

#include <assert.h>

static inline double
create_double(bool s, uint64_t m, uint16_t e)
{
	uint64_t i = 0;
	if (s) {
		i = (1ULL << 63) | (((uint64_t)e) << 52) | (m & 0xFFFFFFFFFFFFFL);
	} else {
		i = (((uint64_t)e) << 52) | (m & 0xFFFFFFFFFFFFFL);
	}

	return *((double *)&i);
}

static int
test_main(void)
{
	double zero = create_double(false, 0, 0);
	assert(jive_double_is_value(zero));
	assert(jive_double_is_zero(zero));
	assert(jive_double_is_unsigned_zero(zero));
	assert(!jive_double_is_signed_zero(zero));
	assert(!jive_double_is_normalized(zero));

	double mzero = create_double(true, 0, 0);
	assert(jive_double_is_value(mzero));
	assert(jive_double_is_zero(mzero));
	assert(!jive_double_is_unsigned_zero(mzero));
	assert(jive_double_is_signed_zero(mzero));
	assert(!jive_double_is_normalized(mzero));

	double nan = create_double(false, 1, 0x7FF);
	assert(!jive_double_is_value(nan));
	assert(jive_double_is_nan(nan));
	assert(!jive_double_is_normalized(nan));

	double inf	= create_double(false, 0, 0x7FF);
	assert(!jive_double_is_value(inf));
	assert(jive_double_is_infinity(inf));
	assert(jive_double_is_unsigned_infinity(inf));
	assert(!jive_double_is_signed_infinity(inf));
	assert(!jive_double_is_normalized(inf));

	double minf = create_double(true, 0, 0x7FF);
	assert(!jive_double_is_value(minf));
	assert(jive_double_is_infinity(minf));
	assert(!jive_double_is_unsigned_infinity(minf));
	assert(jive_double_is_signed_infinity(minf));
	assert(!jive_double_is_normalized(minf));

	assert(jive_double_is_value(6.0));
	assert(!jive_double_is_infinity(6.0));
	assert(!jive_double_is_zero(6.0));
	assert(!jive_double_is_nan(6.0));
	assert(jive_double_is_normalized(6.0));
	assert(jive_double_decoded_mantissa(6.0) == 0x18000000000000);
	assert(jive_double_decoded_exponent(6.0) == 2);

	double denorm = 4.9406564584124654e-324;
	assert(jive_double_is_value(denorm));
	assert(!jive_double_is_infinity(denorm));
	assert(!jive_double_is_zero(denorm));
	assert(!jive_double_is_nan(denorm));
	assert(!jive_double_is_normalized(denorm));
	assert(jive_double_decoded_mantissa(denorm) == 1);
	assert(jive_double_decoded_exponent(denorm) == -1022);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjive/util/test-double", test_main)
