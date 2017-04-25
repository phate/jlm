/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/view.h>
#include <jive/vsdg/graph.h>

static int
verify(const jive::graph * graph)
{
	jive::view(graph->root(), stdout);

	/* FIXME: insert checks when alloca is properly supported */

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-alloca", nullptr, verify)
