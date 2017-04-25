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
verify(const jive::graph * graph)
{
#if 0
	for (jive_node * node : jive::topdown_traverser(const_cast<jive_graph*>(graph))) {
		if (dynamic_cast<const jive::fct::apply_op*>(&node->operation())) {
			assert(node->producer(0)->region == graph->root_region);
		}
	}
#endif
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/tests/test-call_in_loop", nullptr, verify);
