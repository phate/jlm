/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/types/function/fctapply.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/traverser.h>

#include <assert.h>

int
verify()
{
	/* FIXME: insert checks */
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/tests/test-call_in_loop", verify);
