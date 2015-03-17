/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/arch/addresstype.h>
#include <jive/arch/load.h>
#include <jive/arch/memorytype.h>
#include <jive/types/bitstring/type.h>

#include <assert.h>

static int
verify(const jive_graph * graph)
{
	/*FIXME*/
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-load", verify)
