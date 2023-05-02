/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <assert.h>

static void
test_straightening()
{
	using namespace jlm;

	jlm::valuetype vt;
	ipgraph_module module(filepath(""), "", "");

	jlm::cfg cfg(module);
	auto bb1 = basic_block::create(cfg);
	auto bb2 = basic_block::create(cfg);
	auto bb3 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb2->add_outedge(bb3);
	bb3->add_outedge(cfg.exit());

	auto arg = cfg.entry()->append_argument(argument::create("arg", vt));
	bb1->append_last(create_testop_tac({arg}, {&vt}));
	bb2->append_last(create_testop_tac({arg}, {&vt}));
	bb3->append_last(create_testop_tac({arg}, {&vt}));

	auto bb3_last = static_cast<const basic_block*>(bb3)->tacs().last();
	straighten(cfg);

	assert(cfg.nnodes() == 1);
	auto node = cfg.entry()->outedge(0)->sink();

	assert(is<basic_block>(node));
	auto & tacs = static_cast<const basic_block*>(node)->tacs();
	assert(tacs.ntacs() == 3);
	assert(tacs.last() == bb3_last);
}

static void
test_is_structured()
{
	using namespace jlm;

	ipgraph_module module(filepath(""), "", "");

	jlm::cfg cfg(module);
	auto split = basic_block::create(cfg);
	auto bb = basic_block::create(cfg);
	auto join = basic_block::create(cfg);

	cfg.exit()->divert_inedges(split);
	split->add_outedge(join);
	split->add_outedge(bb);
	bb->add_outedge(join);
	join->add_outedge(cfg.exit());

	jlm::print_ascii(cfg, stdout);
	assert(is_structured(cfg));
}

static int
verify()
{
	test_straightening();
	test_is_structured();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-cfg-structure", verify)
