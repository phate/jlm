/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/aggregation.hpp>
#include <jlm/ir/annotation.hpp>
#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/ir/print.hpp>

static bool
contains(
	const jlm::VariableSet & ds,
	const std::vector<const jlm::variable*> & variables)
{
	if (ds.size() != variables.size())
		return false;

	for (const auto & v : variables) {
		if (!ds.contains(v))
			return false;
	}

	return true;
}

static bool
contains(
	const jlm::DemandMap & dm,
	const jlm::aggnode * node,
	const std::vector<const jlm::variable*> & bottom,
	const std::vector<const jlm::variable*> & top,
	const std::vector<const jlm::variable*> & reads,
	const std::vector<const jlm::variable*> & fullwrites,
	const std::vector<const jlm::variable*> & allwrites)
{
	if (dm.find(node) == dm.end())
		return false;

	auto ds = dm.at(node).get();
	return contains(ds->bottom, bottom)
	    && contains(ds->top, top)
	    && contains(ds->reads, reads)
	    && contains(ds->fullwrites, fullwrites);
}

static void
test_block()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	ipgraph_module module(filepath(""), "", "");
	auto v0 = module.create_variable(vt, "v0");

	taclist bb;
	bb.append_last(tac::create(op, {v0}));
	auto v1 = bb.last()->result(0);

	bb.append_last(tac::create(op, {v1}));
	auto v2 = bb.last()->result(0);


	auto root = blockaggnode::create(std::move(bb));

	auto dm = Annotate(*root);
	print(*root, dm, stdout);

	assert(contains(dm, root.get(), {}, {v0}, {v0}, {v1, v2}, {v1, v2}));
}

static void
test_linear()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	ipgraph_module module(filepath(""), "", "");
	argument arg("arg", vt);

	/*
		Setup simple linear CFG: Entry -> B1 -> B2 -> Exit
	*/
	taclist bb1, bb2;
	bb1.append_last(tac::create(op, {&arg}));
	auto v1 = bb1.last()->result(0);

	bb2.append_last(tac::create(op, {v1}));
	auto v2 = bb2.last()->result(0);

	auto en = entryaggnode::create({&arg});
	auto b1 = blockaggnode::create(std::move(bb1));
	auto b2 = blockaggnode::create(std::move(bb2));
	auto xn = exitaggnode::create({v2});
	auto enptr = en.get(), b1ptr = b1.get(), b2ptr = b2.get(), xnptr = xn.get();

	auto l1 = linearaggnode::create(std::move(en), std::move(b1));
	auto l2 = linearaggnode::create(std::move(b2), std::move(xn));
	auto l1ptr = l1.get(), l2ptr = l2.get();

	auto root = linearaggnode::create(std::move(l1), std::move(l2));

	/*
		Create and verify demand map
	*/
	auto dm = Annotate(*root);
	print(*root, dm, stdout);

	assert(contains(dm, xnptr, {}, {v2}, {v2}, {}, {}));
	assert(contains(dm, b2ptr, {v2}, {v1}, {v1}, {v2}, {v2}));
	assert(contains(dm, l2ptr, {}, {v1}, {v1}, {v2}, {v2}));

	assert(contains(dm, b1ptr, {v1}, {&arg}, {&arg}, {v1}, {v1}));
	assert(contains(dm, enptr, {&arg}, {}, {}, {&arg}, {&arg}));
	assert(contains(dm, l1ptr, {v1}, {}, {}, {v1, &arg}, {v1, &arg}));

	assert(contains(dm, root.get(), {}, {}, {}, {v1, &arg, v2}, {v1, &arg, v2}));
}

static void
test_branch()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	ipgraph_module module(filepath(""), "", "");
	auto arg = module.create_variable(vt, "arg");
	auto v3 = module.create_variable(vt, "v3");

	/*
		Setup conditional CFG with nodes bbs, b1, b2, and edges bbs -> b1 and bbs -> b2.
	*/
	taclist bbs, bb1, bb2;
	bbs.append_last(tac::create(op, {arg}));
	auto v1 = bbs.last()->result(0);

	bb2.append_last(tac::create(op, {v1}));
	auto v2 = bb2.last()->result(0);

	bb1.append_last(assignment_op::create(v2, v3));
	bb2.append_last(assignment_op::create(v1, v3));
	bb2.append_last(tac::create(op, {v3}));
	auto v4 = bb2.last()->result(0);

	auto bs = blockaggnode::create(std::move(bbs));
	auto b1 = blockaggnode::create(std::move(bb1));
	auto b2 = blockaggnode::create(std::move(bb2));
	auto bsptr = bs.get(), b1ptr = b1.get(), b2ptr = b2.get();

	auto branch = branchaggnode::create();
	branch->add_child(std::move(b1));
	branch->add_child(std::move(b2));
	auto branchptr = branch.get();

	auto root = linearaggnode::create(std::move(bs), std::move(branch));

	/*
		Create and verify demand map
	*/
	auto dm = Annotate(*root);
	print(*root, dm, stdout);

	assert(contains(dm, b1ptr, {}, {v2}, {v2}, {v3}, {v3}));
	assert(contains(dm, b2ptr, {}, {v1}, {v1}, {v2, v4, v3}, {v2, v4, v3}));
	assert(contains(dm, branchptr, {}, {v1, v2}, {v1, v2}, {v3}, {v2, v3, v4}));
	assert(contains(dm, bsptr, {v1, v2}, {v2, arg}, {arg}, {v1}, {v1}));
	assert(contains(dm, root.get(), {}, {arg, v2}, {arg, v2}, {v1, v3}, {}));
}

static void
test_loop()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	ipgraph_module module(filepath(""), "", "");
	auto v1 = module.create_variable(vt, "v1");
	auto v4 = module.create_variable(vt, "v4");

	taclist bb;
	bb.append_last(tac::create(op, {v1}));
	auto v2 = bb.last()->result(0);

	bb.append_last(tac::create(op, {v2}));
	auto v3 = bb.last()->result(0);

	auto xn = exitaggnode::create({v3, v4});
	auto b = blockaggnode::create(std::move(bb));
	auto xnptr = xn.get(), bptr = b.get();

	auto ln = loopaggnode::create(std::move(b));
	auto lnptr = ln.get();

	auto root = linearaggnode::create(std::move(ln), std::move(xn));

	/*
		Create and verify demand map
	*/
	auto dm = Annotate(*root);
	print(*root, dm, stdout);

	assert(contains(dm, xnptr, {}, {v3, v4}, {v3, v4}, {}, {}));
	assert(contains(dm, bptr, {v1, v3, v4}, {v1, v4}, {v1}, {v2, v3}, {v2, v3}));
	assert(contains(dm, lnptr, {v1, v3, v4}, {v1, v3, v4}, {v1}, {v2, v3}, {v2, v3}));
	assert(contains(dm , root.get(), {}, {v1, v4}, {v1, v4}, {v2, v3}, {v2, v3}));
}

static void
test_branch_in_loop()
{
	using namespace jlm;

	valuetype vt;
	test_op op({&vt}, {&vt});

	ipgraph_module module(filepath(""), "", "");
	auto v1 = module.create_variable(vt, "v1");
	auto v3 = module.create_variable(vt, "v3");

	taclist tl_cb1, tl_cb2;
	tl_cb1.append_last(tac::create(op, {v1}));
	auto v2 = tl_cb1.last()->result(0);

	tl_cb1.append_last(assignment_op::create(v1, v3));
	tl_cb1.append_last(tac::create(op, {v1}));
	auto v4 = tl_cb1.last()->result(0);

	tl_cb2.append_last(assignment_op::create(v1, v3));
	tl_cb2.append_last(assignment_op::create(v4, v3));

	auto xn = exitaggnode::create({v2, v3});
	auto xnptr = xn.get();

	auto b1 = blockaggnode::create(std::move(tl_cb1));
	auto b2 = blockaggnode::create(std::move(tl_cb2));
	auto b1ptr = b1.get(), b2ptr = b2.get();

	auto branch = branchaggnode::create();
	branch->add_child(std::move(b1));
	branch->add_child(std::move(b2));
	auto branchptr = branch.get();

	auto loop = loopaggnode::create(std::move(branch));
	auto loopptr = loop.get();

	auto root = linearaggnode::create(std::move(loop), std::move(xn));

	/*
		Create and verify demand map
	*/
	auto dm = Annotate(*root);
	print(*root, dm, stdout);

	assert(contains(dm, xnptr, {}, {v2, v3}, {v2, v3}, {}, {}));
	assert(contains(dm, b1ptr, {v2, v3, v4}, {v1}, {v1}, {v2, v3, v4}, {v2, v3, v4}));
	assert(contains(dm, b2ptr, {v2, v3, v4}, {v1, v2, v4}, {v1, v4}, {v3}, {v3}));
	assert(contains(dm, branchptr, {v2, v3, v4}, {v1, v2, v4}, {v1, v4}, {v3}, {v2, v3, v4}));
	assert(contains(dm, loopptr, {v1, v2, v3, v4}, {v1, v2, v3, v4}, {v1, v4}, {v3}, {v2, v3, v4}));
	assert(contains(dm, root.get(), {}, {v1, v2, v4}, {v1, v2, v4}, {v3}, {}));
}

static void
test_assignment()
{
	using namespace jlm;

	valuetype vt;

	ipgraph_module module(filepath(""), "", "");
	auto v1 = module.create_variable(vt, "v1");
	auto v2 = module.create_variable(vt, "v2");

	taclist bb;
	bb.append_last(assignment_op::create(v1, v2));

	auto root = blockaggnode::create(std::move(bb));

	/*
		Create and verify demand map
	*/
	auto dm = Annotate(*root);
	print(*root, dm, stdout);

	assert(contains(dm, root.get(), {}, {v1}, {v1}, {v2}, {v2}));
}

static void
test_branch_passby()
{
	using namespace jlm;

	valuetype vt;
	test_op op({}, {&vt});
	ipgraph_module module(filepath(""), "", "");

	auto v3 = module.create_variable(vt, "v3");

	taclist tlsplit, tlb1, tlb2;
	tlsplit.append_last(tac::create(op, {}));
	auto v1 = tlsplit.last()->result(0);

	tlsplit.append_last(tac::create(op, {}));
	auto v2 = tlsplit.last()->result(0);

	tlb1.append_last(assignment_op::create(v1, v2));
	tlb1.append_last(assignment_op::create(v1, v3));
	tlb2.append_last(assignment_op::create(v1, v3));

	auto split = blockaggnode::create(std::move(tlsplit));
	auto branch = branchaggnode::create();
	auto b1 = blockaggnode::create(std::move(tlb1));
	auto b2 = blockaggnode::create(std::move(tlb2));
	auto join = blockaggnode::create();
	auto xn = exitaggnode::create({v1, v2, v3});

	branch->add_child(std::move(b1));
	branch->add_child(std::move(b2));
	auto root = linearaggnode::create(std::move(split), std::move(branch));
	root->add_child(std::move(join));
	root->add_child(std::move(xn));


	auto dm = Annotate(*root);
	print(*root, dm, stdout);

	assert(contains(dm, root.get(), {}, {}, {}, {v1, v2, v3}, {v1, v2, v3}));
	{
		assert(contains(dm, root->child(0), {v1, v2}, {}, {}, {v1, v2}, {v1, v2}));

		auto branch = root->child(1);
		assert(contains(dm, branch, {v2, v3}, {v1, v2}, {v1}, {v3}, {v2, v3}));
		{
			assert(contains(dm, branch->child(0), {v2, v3}, {v1}, {v1}, {v2, v3}, {v2, v3}));
			assert(contains(dm, branch->child(1), {v2, v3}, {v1, v2}, {v1}, {v3}, {v3}));
		}

		assert(contains(dm, root->child(2), {v1, v2, v3}, {v1, v2, v3}, {}, {}, {}));
		assert(contains(dm, root->child(3), {}, {v1, v2, v3}, {v1, v2, v3}, {}, {}));
	}
}

static int
test()
{

	test_block();
	test_linear();
	test_branch();
	test_loop();
	test_branch_in_loop();
	test_assignment();
	test_branch_passby();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/test-annotation", test)
