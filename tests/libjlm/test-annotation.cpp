/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/jlm/ir/aggregation.hpp>
#include <jlm/jlm/ir/annotation.hpp>
#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>
#include <jlm/jlm/ir/view.hpp>

static inline bool
has_variables(
	const jlm::variableset & ds,
	const std::vector<const jlm::variable*> & variables)
{
	if (ds.size() != variables.size())
		return false;

	for (const auto & v : variables) {
		if (ds.find(v) == ds.end())
			return false;
	}

	return true;
}

static inline bool
has_node_and_variables(
	const jlm::aggnode * node,
	const jlm::demandmap & dm,
	const std::vector<const jlm::variable*> & variables)
{
	if (dm.find(node) == dm.end())
		return false;

	auto ds = dm.at(node).get();
	return has_variables(ds->top, variables);
}

static inline void
test_linear_graph()
{
	jlm::module module("", "");

	jlm::cfg cfg(module);
	jlm::valuetype vtype;
	jlm::test_op op({&vtype}, {&vtype});

	auto arg = module.create_variable(vtype, "arg");
	auto v1 = module.create_variable(vtype, false);
	auto v2 = module.create_variable(vtype, false);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb2->add_outedge(cfg.exit_node());

	cfg.entry().append_argument(arg);
	append_last(bb1, create_tac(op, {arg}, {v1}));
	append_last(bb2, create_tac(op, {v1}, {v2}));
	cfg.exit().append_result(v2);

	auto root = jlm::aggregate(cfg);
	jlm::view(*root, stdout);

	auto dm = jlm::annotate(*root);
#if 0
	assert(has_node_and_variables(root.get(), dm, {}));
	{
		auto linear = root->child(0);
		assert(has_node_and_variables(linear, dm, {}));
		{
			auto l = linear->child(0);
			assert(has_node_and_variables(l, dm, {}));
			{
				auto entry = l->child(0);
				assert(has_node_and_variables(entry, dm, {}));

				auto bb1 = l->child(1);
				assert(has_node_and_variables(bb1, dm, {arg}));
			}

			auto bb2 = linear->child(1);
			assert(has_node_and_variables(bb2, dm, {v1}));
		}

		auto exit = root->child(1);
		assert(has_node_and_variables(exit, dm, {v2}));
	}
#endif
}

static void
test_branch_graph()
{
	jlm::module module("", "");

	jlm::cfg cfg(module);
	jlm::valuetype vtype;
	jlm::test_op unop({&vtype}, {&vtype});
	jlm::test_op binop({&vtype, &vtype}, {&vtype});

	auto arg = module.create_variable(vtype, "arg");
	auto v1 = module.create_variable(vtype, "v1");
	auto v2 = module.create_variable(vtype, "v2");
	auto v3 = module.create_variable(vtype, "v3");
	auto v4 = module.create_variable(vtype, "v4");
	auto split = create_basic_block_node(&cfg);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);
	auto join = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(split);
	split->add_outedge(bb1);
	split->add_outedge(bb2);
	bb1->add_outedge(join);
	bb2->add_outedge(join);
	join->add_outedge(cfg.exit_node());

	cfg.entry().append_argument(arg);
	append_last(split, create_tac(unop, {arg}, {v1}));
	append_last(bb1, create_tac(unop, {v1}, {v2}));
	append_last(bb2, create_tac(unop, {v1}, {v3}));
	append_last(join, create_tac(binop, {v2,v3}, {v4}));
	cfg.exit().append_result(v4);

	auto root = jlm::aggregate(cfg);
	jlm::view(*root, stdout);

	auto dm = jlm::annotate(*root);
#if 0
	assert(has_node_and_variables(root.get(), dm, {v2, v3}));
	{
		auto linear = root->child(0);
		assert(has_node_and_variables(linear, dm, {v2, v3}));
		{
			auto branch = linear->child(0);
			assert(has_node_and_variables(branch, dm, {v2, v3}));
			{
				auto linear = branch->child(0);
				assert(has_node_and_variables(linear, dm, {v2, v3}));
				{
					auto entry = linear->child(0);
					assert(has_node_and_variables(entry, dm, {v2, v3}));

					auto split = linear->child(1);
					assert(has_node_and_variables(split, dm, {arg, v2, v3}));
				}

				auto bb1 = branch->child(1);
				assert(has_node_and_variables(bb1, dm, {v1, v3}));

				auto bb2 = branch->child(2);
				assert(has_node_and_variables(bb2, dm, {v1, v2}));
			}

			auto join = linear->child(1);
			assert(has_node_and_variables(join, dm, {v2, v3}));
		}

		auto exit = root->child(1);
		assert(has_node_and_variables(exit, dm, {v4}));
	}
#endif
}

static void
test_loop_graph()
{
	jlm::module module("", "");

	jlm::cfg cfg(module);
	jlm::valuetype vtype;
	jlm::test_op binop({&vtype, &vtype}, {&vtype});

	auto arg = module.create_variable(vtype, "arg");
	auto r = module.create_variable(vtype, "r");
	auto bb = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(bb);
	bb->add_outedge(cfg.exit_node());
	bb->add_outedge(bb);

	cfg.entry().append_argument(arg);
	append_last(bb, create_tac(binop, {arg, r}, {r}));
	cfg.exit().append_result(r);

	auto root = jlm::aggregate(cfg);
	jlm::view(*root, stdout);

	auto dm = jlm::annotate(*root);
#if 0
	assert(has_node_and_variables(root.get(), dm, {r}));
	{
		/* entry */
		assert(has_node_and_variables(root->child(0), dm, {r}));

		/* linear */
		auto linear = root->child(1);
		assert(has_node_and_variables(linear, dm, {arg, r}));
		{
			/* loop */
			auto loop = linear->child(0);
			assert(has_node_and_variables(loop, dm, {arg, r}));
			{
				/* bb */
				assert(has_node_and_variables(loop->child(0), dm, {arg, r}));
			}

			/* exit */
			assert(has_node_and_variables(linear->child(1), dm, {r}));
		}
	}
#endif
}

static int
test_assignment()
{
	jlm::module module("", "");

	jlm::cfg cfg(module);
	jlm::valuetype vtype;

	auto arg = module.create_variable(vtype, "arg", false);
	auto r = module.create_variable(vtype, "result", false);

	auto bb = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(bb);
	bb->add_outedge(cfg.exit_node());

	cfg.entry().append_argument(arg);
	append_last(bb, jlm::create_assignment(vtype, arg, r));
	cfg.exit().append_result(r);

	auto root = jlm::aggregate(cfg);
	auto dm = jlm::annotate(*root);
	jlm::view(*root, dm, stdout);

	assert(dm[root.get()]->top.empty());

	return 0;
}

static int
test()
{
	/* FIXME: avoid aggregation function and build aggregated tree directly */
	test_linear_graph();
	test_branch_graph();
	test_loop_graph();
	test_assignment();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-annotation", test);
