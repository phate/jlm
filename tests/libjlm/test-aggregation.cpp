/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/IR/aggregation/aggregation.hpp>
#include <jlm/IR/aggregation/node.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/module.hpp>

static inline bool
is_entry(jlm::agg::node * node)
{
	return is_entry_structure(node->structure()) && node->nchildren() == 0;
}

static inline bool
is_exit(jlm::agg::node * node)
{
	return is_exit_structure(node->structure()) && node->nchildren() == 0;
}

static inline bool
is_block(jlm::agg::node * node)
{
	return is_block_structure(node->structure()) && node->nchildren() == 0;
}

static inline bool
is_linear(jlm::agg::node * node)
{
	return is_linear_structure(node->structure()) && node->nchildren() == 2;
}

static inline bool
is_loop(jlm::agg::node * node)
{
	return is_loop_structure(node->structure()) && node->nchildren() == 1;
}

static inline bool
is_branch(jlm::agg::node * node, size_t nchildren)
{
	return is_branch_structure(node->structure()) && node->nchildren() == nchildren;
}

static void
test_linear_reduction()
{
	jlm::module module;

	jlm::cfg cfg(module);
	auto bb = create_basic_block_node(&cfg);
	cfg.exit_node()->divert_inedges(bb);
	bb->add_outedge(cfg.exit_node(), 0);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	assert(is_linear(root.get()));
	{
		assert(is_linear(root->child(0)));
		{
			assert(is_entry(root->child(0)->child(0)));
			assert(is_block(root->child(0)->child(1)));
		}

		assert(is_exit(root->child(1)));
	}
}

static void
test_loop_reduction()
{
	jlm::module module;

	jlm::cfg cfg(module);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	cfg.exit_node()->divert_inedges(bb1);
	bb1->add_outedge(bb2, 0);
	bb2->add_outedge(cfg.exit_node(), 0);
	bb2->add_outedge(bb1, 1);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	assert(is_linear(root.get()));
	{
		assert(is_entry(root->child(0)));

		auto linear = root->child(1);
		assert(is_linear(linear));
		{
			auto loop = linear->child(0);
			assert(is_loop(loop));
			{
				auto linear = loop->child(0);
				assert(is_linear(linear));
				{
					assert(is_block(linear->child(0)));
					assert(is_block(linear->child(1)));
				}
			}

			assert(is_exit(linear->child(1)));
		}
	}
}

static void
test_branch_reduction()
{
	jlm::module module;

	jlm::cfg cfg(module);
	auto split = create_basic_block_node(&cfg);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto bb3 = create_basic_block_node(&cfg);
	auto bb4 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(bb1, 0);
	split->add_outedge(bb3, 1);
	bb1->add_outedge(bb2, 0);
	bb2->add_outedge(join, 0);
	bb3->add_outedge(bb4, 0);
	bb4->add_outedge(join, 0);
	join->add_outedge(cfg.exit_node(), 0);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	assert(is_linear(root.get()));
	{
		assert(is_entry(root->child(0)));

		auto linear = root->child(1);
		assert(is_linear(linear));
		{
			auto branch = linear->child(0);
			assert(is_branch(branch, 2));
			{
				auto linear = branch->child(0);
				assert(is_linear(linear));
				{
					assert(is_block(linear->child(0)));
					assert(is_block(linear->child(1)));
				}

				linear = branch->child(1);
				assert(is_linear(linear));
				{
					assert(is_block(linear->child(0)));
					assert(is_block(linear->child(1)));
				}
			}

			assert(is_exit(linear->child(1)));
		}
	}
}

static void
test_branch_loop_reduction()
{
	jlm::module module;

	jlm::cfg cfg(module);
	auto split = create_basic_block_node(&cfg);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto bb3 = create_basic_block_node(&cfg);
	auto bb4 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(bb1, 0);
	split->add_outedge(bb3, 1);
	bb1->add_outedge(bb2, 0);
	bb2->add_outedge(join, 0);
	bb2->add_outedge(bb1, 1);
	bb3->add_outedge(bb4, 0);
	bb4->add_outedge(join, 0);
	bb4->add_outedge(bb3, 1);
	join->add_outedge(cfg.exit_node(), 0);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	assert(is_linear(root.get()));
	{

		assert(is_entry(root->child(0)));

		auto linear = root->child(1);
		assert(is_linear(linear));
		{
			auto branch = linear->child(0);
			assert(is_branch(branch, 2));
			{
				auto loop = branch->child(0);
				assert(is_loop(loop));
				{
					auto linear = loop->child(0);
					assert(is_linear(linear));
					{
						assert(is_block(linear->child(0)));
						assert(is_block(linear->child(1)));
					}
				}

				loop = branch->child(1);
				assert(is_loop(loop));
				{
					auto linear = loop->child(0);
					assert(is_linear(linear));
					{
						assert(is_block(linear->child(0)));
						assert(is_block(linear->child(1)));
					}
				}
			}

			assert(is_exit(linear->child(1)));
		}
	}
}

static void
test_loop_branch_reduction()
{
	jlm::module module;

	jlm::cfg cfg(module);
	auto split = create_basic_block_node(&cfg);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);
	auto bb3 = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(bb1, 0);
	split->add_outedge(bb2, 1);
	bb1->add_outedge(join, 0);
	bb2->add_outedge(join, 0);
	join->add_outedge(bb3, 0);
	bb3->add_outedge(cfg.exit_node(), 0);
	bb3->add_outedge(split, 1);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	assert(is_linear(root.get()));
	{
		assert(is_entry(root->child(0)));

		auto linear = root->child(1);
		assert(is_linear(linear));
		{
			auto loop = linear->child(0);
			assert(is_loop(loop));
			{
				auto linear = loop->child(0);
				assert(is_linear(linear));
				{
					auto branch = linear->child(0);
					assert(is_branch(branch, 2));
					{
						assert(is_block(branch->child(0)));
						assert(is_block(branch->child(1)));
					}
				}

				is_block(linear->child(1));
			}

			is_exit(linear->child(1));
		}
	}
}

static int
test(const jive::graph * graph)
{
	test_linear_reduction();
	test_loop_reduction();
	test_branch_reduction();
	test_branch_loop_reduction();
	test_loop_branch_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-aggregation", nullptr, test);
