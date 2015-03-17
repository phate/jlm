/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/types/bitstring/type.h>

#include <assert.h>

static int
verify(const jive_graph * graph)
{
	/*FIXME*/
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-phi", verify)
