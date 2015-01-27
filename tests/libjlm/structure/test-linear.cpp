/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/frontend/clg.hpp>

#include <assert.h>

static int
verify(jlm::frontend::clg & clg)
{
	assert(clg.nnodes() == 1);

	jlm::frontend::clg_node * node = clg.lookup_function("linear");
	assert(node != nullptr);

	jlm::frontend::cfg * cfg = node->cfg();
//	jive_cfg_view(cfg);

	assert(cfg->nnodes() == 3);
	assert(cfg->is_linear());

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/structure/test-linear", verify);
