/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/frontend/clg.h>

#include <assert.h>

static int
verify(jive::frontend::clg & clg)
{
	assert(clg.nnodes() == 1);

	jive::frontend::clg_node * node = clg.lookup_function("test_while");
	assert(node != nullptr);

	jive::frontend::cfg * cfg = node->cfg();
//	jive_cfg_view(cfg);

	assert(cfg->nnodes() == 5);
	assert(cfg->is_reducible());

	jive::frontend::cfg_node * while_body = cfg->enter()->outedges()[0]->sink()->outedges()[0]->sink();
	assert(while_body->outedges().size() == 2);

	jive::frontend::cfg_node * taken_successor;
	if (while_body->outedges()[0]->index() == 1)
		taken_successor = while_body->outedges()[0]->sink();
	else {
		assert(while_body->outedges()[1]->index() == 1);
		taken_successor = while_body->outedges()[1]->sink();
	}
	assert(taken_successor == while_body);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/structure/test-while", verify);
