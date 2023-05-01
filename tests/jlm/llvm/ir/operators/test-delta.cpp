/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-types.hpp>
#include <test-registry.hpp>

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>

static int
test()
{
	using namespace jlm;

	/* setup graph */

	valuetype vt;
	PointerType pt;
	RvsdgModule rm(filepath(""), "", "");

	auto imp = rm.Rvsdg().add_import({vt, ""});

	auto delta1 = delta::node::Create(
    rm.Rvsdg().root(),
    vt,
		"test-delta1",
		linkage::external_linkage,
    "",
		true);
	auto dep = delta1->add_ctxvar(imp);
	auto d1 = delta1->finalize(create_testop(delta1->subregion(), {dep}, {&vt})[0]);

	auto delta2 = delta::node::Create(
    rm.Rvsdg().root(),
    vt,
		"test-delta2",
		linkage::internal_linkage,
    "",
		false);
	auto d2 = delta2->finalize(create_testop(delta2->subregion(), {}, {&vt})[0]);

  rm.Rvsdg().add_export(d1, {d1->type(), ""});
  rm.Rvsdg().add_export(d2, {d2->type(), ""});

	jive::view(rm.Rvsdg(), stdout);

	/* verify graph */

	assert(rm.Rvsdg().root()->nnodes() == 2);

	assert(delta1->linkage() == linkage::external_linkage);
	assert(delta1->constant() == true);
	assert(delta1->type() == vt);

	assert(delta2->linkage() == linkage::internal_linkage);
	assert(delta2->constant() == false);
	assert(delta2->type() == vt);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/test-delta", test)
