/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <assert.h>

int main(int argc, char ** argv)
{
	assert(argc == 2);
	return jlm::run_unit_test(argv[1]);
}
