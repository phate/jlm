/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-types.hpp>
#include <test-registry.hpp>

#include <jive/view.hpp>

#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/ir/operators/delta.hpp>

static int
test()
{
	using namespace jlm;

	/* setup graph */

	valuetype vt;
	ptrtype pt(vt);
	rvsdg_module rm(filepath(""), "", "");

	auto imp = rm.graph()->add_import({vt, ""});

	auto delta1 = delta::node::create(
		rm.graph()->root(),
		pt,
		"test-delta1",
		linkage::external_linkage,
		true);
	auto dep = delta1->add_ctxvar(imp);
	auto d1 = delta1->finalize(create_testop(delta1->subregion(), {dep}, {&vt})[0]);

	auto delta2 = delta::node::create(
		rm.graph()->root(),
		pt,
		"test-delta2",
		linkage::internal_linkage,
		false);
	auto d2 = delta2->finalize(create_testop(delta2->subregion(), {}, {&vt})[0]);

	rm.graph()->add_export(d1, {d1->type(), ""});
	rm.graph()->add_export(d2, {d2->type(), ""});

	jive::view(*rm.graph(), stdout);

	/* verify graph */

	assert(rm.graph()->root()->nnodes() == 2);

	assert(delta1->linkage() == linkage::external_linkage);
	assert(delta1->constant() == true);
	assert(delta1->type() == pt);

	assert(delta2->linkage() == linkage::internal_linkage);
	assert(delta2->constant() == false);
	assert(delta2->type() == pt);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/operators/test-delta", test)
