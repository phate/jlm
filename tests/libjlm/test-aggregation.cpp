/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/IR/aggregation/aggregation.hpp>
#include <jlm/IR/aggregation/node.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>

static void
test_linear_reduction()
{
	jlm::cfg cfg;
	auto bb = create_basic_block_node(&cfg);
	cfg.exit_node()->divert_inedges(bb);
	bb->add_outedge(cfg.exit_node(), 0);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	assert(is_linear_structure(root->structure()));
	assert(root->nchildren() == 2);

	assert(is_entry_structure(root->child(0)->structure()));
	assert(root->child(0)->nchildren() == 0);

	assert(is_linear_structure(root->child(1)->structure()));
	assert(root->child(1)->nchildren() == 2);

	assert(is_block_structure(root->child(1)->child(0)->structure()));
	assert(root->child(1)->child(0)->nchildren() == 0);

	assert(is_exit_structure(root->child(1)->child(1)->structure()));
	assert(root->child(1)->child(0)->nchildren() == 0);
}

static void
test_loop_reduction()
{
	jlm::cfg cfg;
	auto bb = create_basic_block_node(&cfg);
	cfg.exit_node()->divert_inedges(bb);
	bb->add_outedge(cfg.exit_node(), 0);
	bb->add_outedge(bb, 1);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	assert(is_linear_structure(root->structure()));
	assert(root->nchildren() == 2);

	assert(is_entry_structure(root->child(0)->structure()));
	assert(root->child(0)->nchildren() == 0);

	assert(is_linear_structure(root->child(1)->structure()));
	assert(root->child(1)->nchildren() == 2);

	assert(is_loop_structure(root->child(1)->child(0)->structure()));
	assert(root->child(1)->child(0)->nchildren() == 1);

	assert(is_block_structure(root->child(1)->child(0)->child(0)->structure()));
	assert(root->child(1)->child(0)->child(0)->nchildren() == 0);

	assert(is_exit_structure(root->child(1)->child(1)->structure()));
	assert(root->child(1)->child(1)->nchildren() == 0);
}

static void
test_branch_reduction()
{
	jlm::cfg cfg;
	auto split = create_basic_block_node(&cfg);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(bb1, 0);
	split->add_outedge(bb2, 1);
	bb1->add_outedge(join, 0);
	bb2->add_outedge(join, 0);
	join->add_outedge(cfg.exit_node(), 0);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	assert(is_linear_structure(root->structure()));
	assert(root->nchildren() == 2);

	assert(is_entry_structure(root->child(0)->structure()));
	assert(root->child(0)->nchildren() == 0);

	assert(is_linear_structure(root->child(1)->structure()));
	assert(root->child(1)->nchildren() == 2);

	assert(is_branch_structure(root->child(1)->child(0)->structure()));
	assert(root->child(1)->child(0)->nchildren() == 2);

	assert(is_block_structure(root->child(1)->child(0)->child(0)->structure()));
	assert(root->child(1)->child(0)->child(0)->nchildren() == 0);

	assert(is_block_structure(root->child(1)->child(0)->child(1)->structure()));
	assert(root->child(1)->child(0)->child(1)->nchildren() == 0);

	assert(is_exit_structure(root->child(1)->child(1)->structure()));
	assert(root->child(1)->child(1)->nchildren() == 0);
}

static int
test(const jive::graph * graph)
{
	test_linear_reduction();
	test_loop_reduction();
	test_branch_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-aggregation", nullptr, test);
