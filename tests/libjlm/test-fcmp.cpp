/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <assert.h>

static int
verify(const jive::graph * graph)
{
	jive::view(graph->root(), stdout);

	/* FIXME: add checks and tests for all fcmp operations and types in the test-fcmp.ll */
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-fcmp", nullptr, verify);
