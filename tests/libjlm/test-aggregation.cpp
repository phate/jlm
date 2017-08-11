/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/ir/aggregation/aggregation.hpp>
#include <jlm/ir/aggregation/node.hpp>
#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/module.hpp>

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

static inline void
test_linear_reduction()
{
	jlm::module module("");

	jlm::cfg cfg(module);
	auto bb = create_basic_block_node(&cfg);
	cfg.exit_node()->divert_inedges(bb);
	bb->add_outedge(cfg.exit_node());

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);
#if 0
	assert(is_linear(root.get()));
	{
		assert(is_linear(root->child(0)));
		{
			assert(is_entry(root->child(0)->child(0)));
			assert(is_block(root->child(0)->child(1)));
		}

		assert(is_exit(root->child(1)));
	}
#endif
}

static inline void
test_loop_reduction()
{
	jlm::module module("");

	jlm::cfg cfg(module);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	cfg.exit_node()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb2->add_outedge(cfg.exit_node());
	bb2->add_outedge(bb1);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);
#if 0
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
#endif
}

static void
test_branch_reduction()
{
	jlm::module module("");

	jlm::cfg cfg(module);
	auto split = create_basic_block_node(&cfg);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto bb3 = create_basic_block_node(&cfg);
	auto bb4 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(bb1);
	split->add_outedge(bb3);
	bb1->add_outedge(bb2);
	bb2->add_outedge(join);
	bb3->add_outedge(bb4);
	bb4->add_outedge(join);
	join->add_outedge(cfg.exit_node());

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);
#if 0
	assert(is_linear(root.get()));
	{
		auto branch = root->child(0);
		assert(is_branch(branch, 3));
		{
			auto linear = branch->child(0);
			{
				assert(is_entry(linear->child(0)));
				assert(is_block(linear->child(1)));
			}

			linear = branch->child(1);
			{
				assert(is_block(linear->child(0)));
				assert(is_block(linear->child(1)));
			}

			linear = branch->child(2);
			{
				assert(is_block(linear->child(0)));
				assert(is_block(linear->child(1)));
			}
		}

		auto linear = root->child(1);
		assert(is_linear(linear));
		{
			assert(is_block(linear->child(0)));
			assert(is_exit(linear->child(1)));
		}
	}
#endif
}

static void
test_branch_loop_reduction()
{
	jlm::module module("");

	jlm::cfg cfg(module);
	auto split = create_basic_block_node(&cfg);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto bb3 = create_basic_block_node(&cfg);
	auto bb4 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(bb1);
	split->add_outedge(bb3);
	bb1->add_outedge(bb2);
	bb2->add_outedge(join);
	bb2->add_outedge(bb1);
	bb3->add_outedge(bb4);
	bb4->add_outedge(join);
	bb4->add_outedge(bb3);
	join->add_outedge(cfg.exit_node());

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);
#if 0
	assert(is_linear(root.get()));
	{
		auto branch = root->child(0);
		assert(is_branch(branch, 3));
		{
			auto linear = branch->child(0);
			assert(is_linear(linear));
			{
				assert(is_entry(linear->child(0)));
				assert(is_block(linear->child(1)));
			}

			auto loop = branch->child(1);
			assert(is_loop(loop));
			{
				auto linear = loop->child(0);
				assert(is_linear(linear));
				{
					assert(is_block(linear->child(0)));
					assert(is_block(linear->child(1)));
				}
			}

			loop = branch->child(2);
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

		auto linear = root->child(1);
		assert(is_linear(linear));
		{
			assert(is_block(linear->child(0)));
			assert(is_exit(linear->child(1)));
		}
	}
#endif
}

static void
test_loop_branch_reduction()
{
	jlm::module module("");

	jlm::cfg cfg(module);
	auto split = create_basic_block_node(&cfg);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);
	auto bb3 = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(bb1);
	split->add_outedge(bb2);
	bb1->add_outedge(join);
	bb2->add_outedge(join);
	join->add_outedge(bb3);
	bb3->add_outedge(cfg.exit_node());
	bb3->add_outedge(split);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);
#if 0
	assert(is_linear(root.get()));
	{
		assert(is_entry(root->child(0)));

		auto linear = root->child(1);
		assert(is_linear(linear));
		{
			auto loop = linear->child(0);
			{
				auto linear = loop->child(0);
				assert(is_linear(linear));
				{
					auto l = linear->child(0);
					assert(is_linear(l));
					{
						auto branch = l->child(0);
						assert(is_branch(branch, 3));
						{
							assert(is_block(branch->child(0)));
							assert(is_block(branch->child(1)));
							assert(is_block(branch->child(2)));
						}

						assert(is_block(l->child(1)));
					}

					assert(is_block(linear->child(1)));
				}

			}

			assert(is_exit(linear->child(1)));
		}
	}
#endif
}

static void
test_ifthen_reduction()
{
	jlm::module module("");

	jlm::cfg cfg(module);
	auto split = create_basic_block_node(&cfg);
	auto n2 = create_basic_block_node(&cfg);
	auto n3 = create_basic_block_node(&cfg);
	auto n4 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(n4);
	split->add_outedge(n2);
	n2->add_outedge(n3);
	n3->add_outedge(join);
	n4->add_outedge(join);
	join->add_outedge(cfg.exit_node());

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);
}

static int
test()
{
	/* FIXME: re-activate asserts */
	test_linear_reduction();
	test_loop_reduction();
	test_branch_reduction();
	test_branch_loop_reduction();
	test_loop_branch_reduction();
	test_ifthen_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-aggregation", test);
