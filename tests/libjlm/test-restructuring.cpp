/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/jlm2rvsdg/restructuring.hpp>

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
	bb1->add_outedge(bb2);
	bb2->add_outedge(bb2);
	bb2->add_outedge(bb3);
	bb3->add_outedge(bb1);
	bb3->add_outedge(cfg.exit_node());

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
