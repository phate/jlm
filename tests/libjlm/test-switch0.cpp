/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/frontend/clg.h>

#include <assert.h>

static const char * program =
"\
	int switch0(int a) \
	{ \
		switch(a) { \
			case 0: break; \
			case 1: break; \
			default:; \
		} \
		return 0; \
	} \
";

static int
verify(jive::frontend::clg & clg)
{
	assert(clg.nnodes() == 1);

	jive::frontend::clg_node * node = clg.lookup_function("switch0");
	assert(node != nullptr);

	jive::frontend::cfg & cfg = node->cfg();
//	jive_cfg_view(cfg);

	assert(cfg.nnodes() == 7);
	assert(cfg.is_structured());

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-switch0", program, verify);
