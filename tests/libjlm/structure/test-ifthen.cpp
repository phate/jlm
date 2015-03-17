/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <assert.h>

static int
verify(const jive_graph * graph)
{
	/*FIXME*/
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/structure/test-ifthen", verify);
