/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/IR/aggregation/aggregation.hpp>
#include <jlm/IR/aggregation/annotation.hpp>
#include <jlm/IR/aggregation/node.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/module.hpp>

static inline bool
has_variables(
	const jlm::agg::dset & ds,
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

static void
test_linear_graph()
{
	jlm::module module;

	jlm::cfg cfg(module);
	jlm::valuetype vtype;
	jlm::test_op op({&vtype}, {&vtype});

	auto arg = module.create_variable(vtype, "arg");
	auto v1 = module.create_variable(vtype, false);
	auto v2 = module.create_variable(vtype, false);
	auto bb1 = create_basic_block_node(&cfg);
	auto bb2 = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(bb1);
	bb1->add_outedge(bb2, 0);
	bb2->add_outedge(cfg.exit_node(), 0);

	cfg.entry().append_argument(arg);
	static_cast<jlm::basic_block*>(&bb1->attribute())->append(create_tac(op, {arg}, {v1}));
	static_cast<jlm::basic_block*>(&bb2->attribute())->append(create_tac(op, {v1}, {v2}));
	cfg.exit().append_result(v2);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	auto dm = jlm::agg::annotate(*root);

	/* linear */
	assert(dm.find(root.get()) != dm.end());
	auto ds = dm[root.get()].get();
	assert(has_variables(ds->top, {}));

	/* entry */
	assert(dm.find(root->child(0)) != dm.end());
	ds = dm[root->child(0)].get();
	assert(has_variables(ds->top, {}));

	/* linear */
	assert(dm.find(root->child(1)) != dm.end());
	ds = dm[root->child(1)].get();
	assert(has_variables(ds->top, {arg}));

	/* bb1 */
	assert(dm.find(root->child(1)->child(0)) != dm.end());
	ds = dm[root->child(1)->child(0)].get();
	assert(has_variables(ds->top, {arg}));

	/* linear */
	assert(dm.find(root->child(1)->child(1)) != dm.end());
	ds = dm[root->child(1)->child(1)].get();
	assert(has_variables(ds->top, {v1}));

	/* bb2 */
	assert(dm.find(root->child(1)->child(1)->child(0)) != dm.end());
	ds = dm[root->child(1)->child(1)->child(0)].get();
	assert(has_variables(ds->top, {v1}));

	/* exit */
	assert(dm.find(root->child(1)->child(1)->child(1)) != dm.end());
	ds = dm[root->child(1)->child(1)->child(1)].get();
	assert(has_variables(ds->top, {v2}));
}

static void
test_branch_graph()
{
	jlm::module module;

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
	split->add_outedge(bb1, 0);
	split->add_outedge(bb2, 1);
	bb1->add_outedge(join, 0);
	bb2->add_outedge(join, 0);
	join->add_outedge(cfg.exit_node(), 0);

	cfg.entry().append_argument(arg);
	static_cast<jlm::basic_block*>(&split->attribute())->append(create_tac(unop, {arg}, {v1}));
	static_cast<jlm::basic_block*>(&bb1->attribute())->append(create_tac(unop, {v1}, {v2}));
	static_cast<jlm::basic_block*>(&bb2->attribute())->append(create_tac(unop, {v1}, {v3}));
	static_cast<jlm::basic_block*>(&join->attribute())->append(create_tac(binop, {v2, v3}, {v4}));
	cfg.exit().append_result(v4);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	auto dm = jlm::agg::annotate(*root);

	/* linear */
	assert(dm.find(root.get()) != dm.end());
	auto ds = dm[root.get()].get();
	assert(has_variables(ds->top, {v2, v3}));

	/* entry */
	assert(dm.find(root->child(0)) != dm.end());
	ds = dm[root->child(0)].get();
	assert(has_variables(ds->top, {v2, v3}));

	/* linear */
	assert(dm.find(root->child(1)) != dm.end());
	ds = dm[root->child(1)].get();
	assert(has_variables(ds->top, {arg, v2, v3}));

	/* branch */
	assert(dm.find(root->child(1)->child(0)) != dm.end());
	ds = dm[root->child(1)->child(0)].get();
	assert(has_variables(ds->top, {arg, v2, v3}));

	/* bb2 */
	assert(dm.find(root->child(1)->child(0)->child(0)) != dm.end());
	ds = dm[root->child(1)->child(0)->child(0)].get();
	assert(has_variables(ds->top, {v1, v2}));

	/* bb1 */
	assert(dm.find(root->child(1)->child(0)->child(1)) != dm.end());
	ds = dm[root->child(1)->child(0)->child(1)].get();
	assert(has_variables(ds->top, {v1, v3}));

	/* exit */
	assert(dm.find(root->child(1)->child(1)) != dm.end());
	ds = dm[root->child(1)->child(1)].get();
	assert(has_variables(ds->top, {v4}));
}

static void
test_loop_graph()
{
	jlm::module module;

	jlm::cfg cfg(module);
	jlm::valuetype vtype;
	jlm::test_op binop({&vtype, &vtype}, {&vtype});

	auto arg = module.create_variable(vtype, "arg");
	auto r = module.create_variable(vtype, "r");
	auto bb = create_basic_block_node(&cfg);

	cfg.exit_node()->divert_inedges(bb);
	bb->add_outedge(cfg.exit_node(), 0);
	bb->add_outedge(bb, 1);

	cfg.entry().append_argument(arg);
	static_cast<jlm::basic_block*>(&bb->attribute())->append(create_tac(binop, {arg, r}, {r}));
	cfg.exit().append_result(r);

	auto root = jlm::agg::aggregate(cfg);
	jlm::agg::view(*root, stdout);

	auto dm = jlm::agg::annotate(*root);

	/* linear */
	assert(dm.find(root.get()) != dm.end());
	auto ds = dm[root.get()].get();
	assert(has_variables(ds->top, {r}));

	/* entry */
	assert(dm.find(root->child(0)) != dm.end());
	ds = dm[root->child(0)].get();
	assert(has_variables(ds->top, {r}));

	/* linear */
	assert(dm.find(root->child(1)) != dm.end());
	ds = dm[root->child(1)].get();
	assert(has_variables(ds->top, {arg, r}));

	/* loop */
	assert(dm.find(root->child(1)->child(0)) != dm.end());
	ds = dm[root->child(1)->child(0)].get();
	assert(has_variables(ds->top, {arg, r}));

	/* bb2 */
	assert(dm.find(root->child(1)->child(0)->child(0)) != dm.end());
	ds = dm[root->child(1)->child(0)->child(0)].get();
	assert(has_variables(ds->top, {arg, r}));

	/* exit */
	assert(dm.find(root->child(1)->child(1)) != dm.end());
	ds = dm[root->child(1)->child(1)].get();
	assert(has_variables(ds->top, {r}));
}

static int
test(const jive::graph * graph)
{
	/* FIXME: avoid aggregation function and build aggregated tree directly */
	test_linear_graph();
	test_branch_graph();
	test_loop_graph();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-annotation", nullptr, test);
