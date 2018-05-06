/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/ir/view.hpp>

#include <assert.h>

static void
test_straightening()
{
	jlm::valuetype vt;
	jlm::module module("", "");
	auto v = module.create_variable(vt, "v", false);

	jlm::cfg cfg(module);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto bb3 = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb2->add_outedge(bb3);
	bb3->add_outedge(cfg.exit_node());

	jlm::append_last(bb1, jlm::create_testop_tac({v}, {v}));
	jlm::append_last(bb2, jlm::create_testop_tac({v}, {v}));
	jlm::append_last(bb3, jlm::create_testop_tac({v}, {v}));

	auto bb3_last = static_cast<const jlm::basic_block*>(&bb3->attribute())->last();
	straighten(cfg);

	assert(cfg.nnodes() == 3);
	auto node = cfg.entry_node()->outedge(0)->sink();

	assert(is_basic_block(node->attribute()));
	auto bb = static_cast<const jlm::basic_block*>(&node->attribute());
	assert(bb->ntacs() == 3);
	assert(bb->last() == bb3_last);
}

static void
test_is_structured()
{
	jlm::module module("", "");

	jlm::cfg cfg(module);
	auto split = create_basic_block_node(&cfg);
	auto bb = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(join);
	split->add_outedge(bb);
	bb->add_outedge(join);
	join->add_outedge(cfg.exit_node());

	jlm::view_ascii(cfg, stdout);
	assert(is_structured(cfg));
}

static int
verify()
{
	test_straightening();
	test_is_structured();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-cfg-structure", verify);
