/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/ir/aggregation.hpp>
#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/print.hpp>

static bool
is_entry(const jlm::aggnode * node)
{
	return jlm::is<jlm::entryaggnode>(node) && node->nchildren() == 0;
}

static bool
is_exit(const jlm::aggnode * node)
{
	return jlm::is<jlm::exitaggnode>(node) && node->nchildren() == 0;
}

static bool
is_block(const jlm::aggnode * node)
{
	return jlm::is<jlm::blockaggnode>(node) && node->nchildren() == 0;
}

static bool
is_linear(const jlm::aggnode * node, size_t nchildren)
{
	return jlm::is<jlm::linearaggnode>(node) && node->nchildren() == nchildren;
}

static bool
is_loop(const jlm::aggnode * node)
{
	return jlm::is<jlm::loopaggnode>(node) && node->nchildren() == 1;
}

static bool
is_branch(const jlm::aggnode * node, size_t nchildren)
{
	return jlm::is<jlm::branchaggnode>(node) && node->nchildren() == nchildren;
}

static inline void
test_linear_reduction()
{
	using namespace jlm;

	auto setup_cfg = [](ipgraph_module & module)
	{
		auto cfg = jlm::cfg::create(module);

		auto bb = basic_block::create(*cfg);
		cfg->exit()->divert_inedges(bb);
		bb->add_outedge(cfg->exit());

		return cfg;
	};

	auto verify_aggtree = [](const aggnode & root)
	{
		assert(is_linear(&root, 3));
		{
			assert(is_entry(root.child(0)));
			assert(is_block(root.child(1)));
			assert(is_exit(root.child(2)));
		}
	};

	ipgraph_module module(filepath(""), "", "");
	auto cfg = setup_cfg(module);

	auto root = jlm::aggregate(*cfg);
	aggnode::normalize(*root);
	jlm::print(*root, stdout);

	verify_aggtree(*root);
}

static inline void
test_loop_reduction()
{
	using namespace jlm;

	auto setup_cfg = [](ipgraph_module & module)
	{
		auto cfg = jlm::cfg::create(module);

		auto bb1 = basic_block::create(*cfg);
		auto bb2 = basic_block::create(*cfg);

		cfg->exit()->divert_inedges(bb1);
		bb1->add_outedge(bb2);
		bb2->add_outedge(cfg->exit());
		bb2->add_outedge(bb1);

		return cfg;
	};

	auto verify_aggtree = [](const aggnode & root)
	{
		assert(is_linear(&root, 3));
		{
			assert(is_entry(root.child(0)));

			auto loop = root.child(1);
			assert(is_loop(loop));
			{
				auto linear = loop->child(0);
				assert(is_linear(linear, 2));
				{
					assert(is_block(linear->child(0)));
					assert(is_block(linear->child(1)));
				}
			}

			assert(is_exit(root.child(2)));
		}
	};

	ipgraph_module module(filepath(""), "", "");
	auto cfg = setup_cfg(module);

	auto root = jlm::aggregate(*cfg);
	aggnode::normalize(*root);
	jlm::print(*root, stdout);

	verify_aggtree(*root);
}

static void
test_branch_reduction()
{
	using namespace jlm;

	auto setup_cfg = [](ipgraph_module & module)
	{
		auto cfg = cfg::create(module);

		auto split = basic_block::create(*cfg);
		auto bb1 = basic_block::create(*cfg);
		auto bb2 = basic_block::create(*cfg);
		auto bb3 = basic_block::create(*cfg);
		auto bb4 = basic_block::create(*cfg);
		auto join = basic_block::create(*cfg);

		cfg->exit()->divert_inedges(split);
		split->add_outedge(bb1);
		split->add_outedge(bb3);
		bb1->add_outedge(bb2);
		bb2->add_outedge(join);
		bb3->add_outedge(bb4);
		bb4->add_outedge(join);
		join->add_outedge(cfg->exit());

		return cfg;
	};

	auto verify_aggtree = [](const aggnode & root)
	{
		assert(is_linear(&root, 5));
		{
			assert(is_entry(root.child(0)));
			assert(is_block(root.child(1)));

			auto branch = root.child(2);
			assert(is_branch(branch, 2));
			{
				auto linear = branch->child(0);
				assert(is_linear(linear, 2));
				{
					assert(is_block(linear->child(0)));
					assert(is_block(linear->child(1)));
				}

				linear = branch->child(1);
				assert(is_linear(linear, 2));
				{
					assert(is_block(linear->child(0)));
					assert(is_block(linear->child(1)));
				}
			}

			assert(is_block(root.child(3)));
			assert(is_exit(root.child(4)));
		}
	};

	ipgraph_module module(filepath(""), "", "");
	auto cfg = setup_cfg(module);

	auto root = jlm::aggregate(*cfg);
	aggnode::normalize(*root);
	jlm::print(*root, stdout);

	verify_aggtree(*root);
}

static void
test_branch_loop_reduction()
{
	using namespace jlm;

	auto setup_cfg = [](ipgraph_module & module)
	{
		auto cfg = cfg::create(module);
		auto split = basic_block::create(*cfg);
		auto bb1 = basic_block::create(*cfg);
		auto bb2 = basic_block::create(*cfg);
		auto bb3 = basic_block::create(*cfg);
		auto bb4 = basic_block::create(*cfg);
		auto join = basic_block::create(*cfg);

		cfg->exit()->divert_inedges(split);
		split->add_outedge(bb1);
		split->add_outedge(bb3);
		bb1->add_outedge(bb2);
		bb2->add_outedge(join);
		bb2->add_outedge(bb1);
		bb3->add_outedge(bb4);
		bb4->add_outedge(join);
		bb4->add_outedge(bb3);
		join->add_outedge(cfg->exit());

		return cfg;
	};

	auto verify_aggtree = [](const aggnode & root)
	{
		assert(is_linear(&root, 5));
		{
			assert(is_entry(root.child(0)));
			assert(is_block(root.child(1)));

			auto branch = root.child(2);
			assert(is_branch(branch, 2));
			{
				auto loop = branch->child(0);
				assert(is_loop(loop));
				{
					auto linear = loop->child(0);
					assert(is_linear(linear, 2));
					{
						assert(is_block(linear->child(0)));
						assert(is_block(linear->child(1)));
					}
				}

				loop = branch->child(1);
				assert(is_loop(loop));
				{
					auto linear = loop->child(0);
					assert(is_linear(linear, 2));
					{
						assert(is_block(linear->child(0)));
						assert(is_block(linear->child(1)));
					}
				}
			}

			assert(is_block(root.child(3)));
			assert(is_exit(root.child(4)));
		}
	};

	ipgraph_module module(filepath(""), "", "");
	auto cfg = setup_cfg(module);

	auto root = jlm::aggregate(*cfg);
	aggnode::normalize(*root);
	jlm::print(*root, stdout);

	verify_aggtree(*root);
}

static void
test_loop_branch_reduction()
{
	using namespace jlm;

	auto setup_cfg = [](ipgraph_module & module)
	{
		auto cfg = cfg::create(module);

		auto split = basic_block::create(*cfg);
		auto bb1 = basic_block::create(*cfg);
		auto bb2 = basic_block::create(*cfg);
		auto join = basic_block::create(*cfg);
		auto bb3 = basic_block::create(*cfg);

		cfg->exit()->divert_inedges(split);
		split->add_outedge(bb1);
		split->add_outedge(bb2);
		bb1->add_outedge(join);
		bb2->add_outedge(join);
		join->add_outedge(bb3);
		bb3->add_outedge(cfg->exit());
		bb3->add_outedge(split);

		return cfg;
	};

	auto verify_aggtree = [](const aggnode & root)
	{
		assert(is_linear(&root, 3));
		{
			assert(is_entry(root.child(0)));

			auto loop = root.child(1);
			assert(is_loop(loop));
			{
				auto linear = loop->child(0);
				assert(is_linear(linear, 4));
				{
					assert(is_block(linear->child(0)));

					auto branch = linear->child(1);
					assert(is_branch(branch, 2));
					{
						assert(is_block(branch->child(0)));
						assert(is_block(branch->child(1)));
					}

					assert(is_block(linear->child(2)));
					assert(is_block(linear->child(3)));
				}
			}

			assert(is_exit(root.child(2)));
		}
	};

	ipgraph_module module(filepath(""), "", "");
	auto cfg = setup_cfg(module);

	auto root = jlm::aggregate(*cfg);
	aggnode::normalize(*root);
	jlm::print(*root, stdout);

	verify_aggtree(*root);
}

static void
test_ifthen_reduction()
{
	using namespace jlm;

	auto setup_cfg = [](ipgraph_module & module)
	{
		auto cfg = cfg::create(module);

		auto split = basic_block::create(*cfg);
		auto bb1 = basic_block::create(*cfg);
		auto bb2 = basic_block::create(*cfg);
		auto bb3 = basic_block::create(*cfg);
		auto join = basic_block::create(*cfg);

		cfg->exit()->divert_inedges(split);
		split->add_outedge(bb3);
		split->add_outedge(bb1);
		bb1->add_outedge(bb2);
		bb2->add_outedge(join);
		bb3->add_outedge(join);
		join->add_outedge(cfg->exit());

		return cfg;
	};

	auto verify_aggtree = [](const aggnode & root)
	{
		assert(is_linear(&root, 5));
		{
			assert(is_entry(root.child(0)));
			assert(is_block(root.child(1)));

			auto branch = root.child(2);
			assert(is_branch(branch, 2));
			{
				assert(is_block(branch->child(0)));

				auto linear = branch->child(1);
				assert(is_linear(linear, 2));
				{
					assert(is_block(linear->child(0)));
					assert(is_block(linear->child(1)));
				}
			}

			assert(is_block(root.child(3)));
			assert(is_exit(root.child(4)));
		}
	};

	ipgraph_module module(filepath(""), "", "");
	auto cfg = setup_cfg(module);

	auto root = jlm::aggregate(*cfg);
	aggnode::normalize(*root);
	jlm::print(*root, stdout);

	verify_aggtree(*root);
}

static int
test()
{
	test_linear_reduction();
	test_loop_reduction();
	test_branch_reduction();
	test_branch_loop_reduction();
	test_loop_branch_reduction();
	test_ifthen_reduction();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/test-aggregation", test)
