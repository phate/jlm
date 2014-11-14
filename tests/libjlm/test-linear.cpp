/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/frontend/clg.h>

#include <assert.h>

static const char * program =
"\
	int linear(int a) \
	{ \
		return 6+7; \
	} \
";

static int
verify(jive::frontend::clg & clg)
{
	assert(clg.nnodes() == 1);

	jive::frontend::clg_node * node = clg.lookup_function("linear");
	assert(node != nullptr);

	jive::frontend::cfg & cfg = node->cfg();
	jive_cfg_view(cfg);

	assert(cfg.nnodes() == 3);
	assert(cfg.is_linear());

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-linear", program, verify);
