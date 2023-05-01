/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/print.hpp>

static void
test_remove_node()
{
	using namespace jlm;

	/* setup cfg */

	ipgraph_module im(filepath(""), "", "");

	jlm::cfg cfg(im);

	auto bb0 = basic_block::create(cfg);
	bb0->add_outedge(bb0);
	bb0->add_outedge(cfg.exit());

	print_ascii(cfg, stdout);

	/* verify inedge diversion */

	cfg.remove_node(bb0);
	assert(cfg.nnodes() == 0);
}

static int
test()
{
	test_remove_node();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-cfg", test)
