/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/destruction/restructuring.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/module.hpp>

#include <assert.h>

static void
test_dowhile()
{
	jlm::module module;

	jlm::cfg cfg(module);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto bb3 = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(bb1);
	bb1->add_outedge(bb2, 0);
	bb2->add_outedge(bb3, 1);
	bb2->add_outedge(bb2, 0);
	bb3->add_outedge(cfg.exit_node(), 1);
	bb3->add_outedge(bb1, 0);

	size_t nnodes = cfg.nnodes();
	restructure(&cfg);
	assert(nnodes == cfg.nnodes());
}

static int
verify()
{
	test_dowhile();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-restructuring", verify);
