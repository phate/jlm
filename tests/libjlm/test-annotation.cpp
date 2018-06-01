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
#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>
#include <jlm/jlm/ir/view.hpp>

static bool
contains(
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

static bool
contains(
	const jlm::demandmap & dm,
	const jlm::aggnode * node,
	const std::vector<const jlm::variable*> & bottom,
	const std::vector<const jlm::variable*> & top,
	const std::vector<const jlm::variable*> & reads,
	const std::vector<const jlm::variable*> & writes)
{
	if (dm.find(node) == dm.end())
		return false;

	auto ds = dm.at(node).get();
	return contains(ds->bottom, bottom)
	    && contains(ds->top, top)
	    && contains(ds->reads, reads)
	    && contains(ds->writes, writes);
}

static void
test_block()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	jlm::module module("", "");
	auto v0 = module.create_variable(vt, "v0", false);
	auto v1 = module.create_variable(vt, "v1", false);
	auto v2 = module.create_variable(vt, "v2", false);

	basic_block bb;
	bb.append_last(create_tac(op, {v0}, {v1}));
	bb.append_last(create_tac(op, {v1}, {v2}));

	auto root = blockaggnode::create(std::move(bb));

	auto dm = annotate(*root);
	view(*root, dm, stdout);

	assert(contains(dm, root.get(), {}, {v0}, {v0}, {v1, v2}));
}

static void
test_linear()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	jlm::module module("", "");
	auto arg = module.create_variable(vt, "arg", false);
	auto v1 = module.create_variable(vt, "v1", false);
	auto v2 = module.create_variable(vt, "v2", false);

	/*
		Setup simple linear CFG: Entry -> B1 -> B2 -> Exit
	*/
	entry ea({arg});
	jlm::exit xa({v2});

	basic_block bb1, bb2;
	bb1.append_last(create_tac(op, {arg}, {v1}));
	bb2.append_last(create_tac(op, {v1}, {v2}));

	auto en = entryaggnode::create(ea);
	auto b1 = blockaggnode::create(std::move(bb1));
	auto b2 = blockaggnode::create(std::move(bb2));
	auto xn = exitaggnode::create(xa);
	auto enptr = en.get(), b1ptr = b1.get(), b2ptr = b2.get(), xnptr = xn.get();

	auto l1 = linearaggnode::create(std::move(en), std::move(b1));
	auto l2 = linearaggnode::create(std::move(b2), std::move(xn));
	auto l1ptr = l1.get(), l2ptr = l2.get();

	auto root = linearaggnode::create(std::move(l1), std::move(l2));

	/*
		Create and verify demand map
	*/
	auto dm = annotate(*root);
	view(*root, dm, stdout);

	assert(contains(dm, xnptr, {}, {v2}, {v2}, {}));
	assert(contains(dm, b2ptr, {v2}, {v1}, {v1}, {v2}));
	assert(contains(dm, l2ptr, {}, {v1}, {v1}, {v2}));

	assert(contains(dm, b1ptr, {v1}, {arg}, {arg}, {v1}));
	assert(contains(dm, enptr, {arg}, {}, {}, {arg}));
	assert(contains(dm, l1ptr, {v1}, {}, {}, {v1, arg}));

	assert(contains(dm, root.get(), {}, {}, {}, {v1, arg, v2}));
}

static void
test_branch()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	jlm::module module("", "");
	auto arg = module.create_variable(vt, "arg", false);
	auto v1 = module.create_variable(vt, "v1", false);
	auto v2 = module.create_variable(vt, "v2", false);
	auto v3 = module.create_variable(vt, "v3", false);
	auto v4 = module.create_variable(vt, "v4", false);

	/*
		Setup conditional CFG with nodes bbs, b1, b2, and edges bbs -> b1 and bbs -> b2.
	*/
	basic_block bbs, bb1, bb2;
	bbs.append_last(create_tac(op, {arg}, {v1}));
	bb1.append_last(create_tac(op, {v2}, {v3}));
	bb2.append_last(create_tac(op, {v1}, {v2}));
	bb2.append_last(create_tac(op, {v1}, {v3}));
	bb2.append_last(create_tac(op, {v3}, {v4}));

	auto bs = blockaggnode::create(std::move(bbs));
	auto b1 = blockaggnode::create(std::move(bb1));
	auto b2 = blockaggnode::create(std::move(bb2));
	auto bsptr = bs.get(), b1ptr = b1.get(), b2ptr = b2.get();

	auto root = branchaggnode::create(std::move(bs));
	root->add_child(std::move(b1));
	root->add_child(std::move(b2));

	/*
		Create and verify demand map
	*/
	auto dm = annotate(*root);
	view(*root, dm, stdout);

	assert(contains(dm, b1ptr, {}, {v2}, {v2}, {v3}));
	assert(contains(dm, b2ptr, {}, {v1}, {v1}, {v2, v4, v3}));
	assert(contains(dm, bsptr, {v1, v2}, {v2, arg}, {arg}, {v1}));
	assert(contains(dm, root.get(), {}, {arg, v2}, {arg, v2}, {v1, v3}));
}

static void
test_loop()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	jlm::module module("", "");
	auto v1 = module.create_variable(vt, "v1", false);
	auto v2 = module.create_variable(vt, "v2", false);
	auto v3 = module.create_variable(vt, "v3", false);
	auto v4 = module.create_variable(vt, "v4", false);

	jlm::exit xa({v3, v4});

	basic_block bb;
	bb.append_last(create_tac(op, {v1}, {v2}));
	bb.append_last(create_tac(op, {v2}, {v3}));

	auto xn = exitaggnode::create(xa);
	auto b = blockaggnode::create(std::move(bb));
	auto xnptr = xn.get(), bptr = b.get();

	auto ln = loopaggnode::create(std::move(b));
	auto lnptr = ln.get();

	auto root = linearaggnode::create(std::move(ln), std::move(xn));

	/*
		Create and verify demand map
	*/
	auto dm = annotate(*root);
	view(*root, dm, stdout);

	assert(contains(dm, xnptr, {}, {v3, v4}, {v3, v4}, {}));
	assert(contains(dm, bptr, {v1, v3}, {v1}, {v1}, {v2, v3}));
	assert(contains(dm, lnptr, {v1, v3}, {v1, v3}, {v1}, {v2, v3}));
	assert(contains(dm , root.get(), {}, {v1, v4}, {v1, v4}, {v2, v3}));
}

static void
test_assignment()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	jlm::module module("", "");
	auto v1 = module.create_variable(vt, "v1", false);
	auto v2 = module.create_variable(vt, "v2", false);

	basic_block bb;
	bb.append_last(create_tac(op, {v1}, {v2}));

	auto root = blockaggnode::create(std::move(bb));

	/*
		Create and verify demand map
	*/
	auto dm = annotate(*root);
	view(*root, dm, stdout);

	assert(contains(dm, root.get(), {}, {v1}, {v1}, {v2}));
}

static int
test()
{
	test_block();
	test_linear();
	test_branch();
	test_loop();
	test_assignment();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-annotation", test);
