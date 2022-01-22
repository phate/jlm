/*
 * Copyright 2013 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <assert.hpp>
#include <stdbool.hpp>
#include <stdlib.hpp>

#include "test-registry.h"

#include <jive/util/heapsort.hpp>

static inline bool
int_compare(int a, int b) {
	return a > b;
}

DECLARE_HEAPSORT_FUNCTIONS(int_heap, int, int_compare);

static void
sort_and_verify(int * vec, size_t size)
{
	int_heap_sort(vec, size);
	size_t n;
	for (n = 1; n < size; ++n) {
		assert(vec[n - 1] <= vec[n]);
	}
}

static int
test_main(void)
{
	int vec[4096];
	size_t n = 0;
	for (n = 0; n < 4096; ++n)
		vec[n] = n;
	sort_and_verify(vec, 4096);
	
	for (n = 0; n < 4096; ++n)
		vec[n] = 4096 - n;
	sort_and_verify(vec, 4096);
	
	for (n = 0; n < 4096; ++n)
		vec[n] = (n&1 << 12) | (n&2 << 5) | (n&4 << 7) | (n&8) |
			 (n&16 << 7) | (n&32 << 3) | (n&64 >> 1) | (n&128 >> 5) |
			 (n&256 << 2) | (n&512 >> 2) | (n&1024 >> 6) | (n&2048 >> 10);
	sort_and_verify(vec, 4096);
	
	return 0;
}

JIVE_UNIT_TEST_REGISTER("util/test-heapsort", test_main);
