/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <jive/rvsdg/phi.hpp>
#include <jive/view.hpp>

#include <jlm/backend/llvm/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators/delta.hpp>
#include <jlm/ir/print.hpp>
#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	ptrtype pt(vt);

	rvsdg_module rm(filepath(""), "", "");

	/* setup graph */
	auto imp = rm.graph()->add_import(impport(pt, "", linkage::external_linkage));

	jive::phi::builder pb;
	pb.begin(rm.graph()->root());
	auto region = pb.subregion();
	auto r1 = pb.add_recvar(pt);
	auto r2 = pb.add_recvar(pt);
	auto dep = pb.add_ctxvar(imp);

	jive::output * delta1, * delta2;
	{
		auto delta = delta::node::create(
			region,
			vt,
			"test-delta1",
			linkage::external_linkage,
			false);
		auto dep1 = delta->add_ctxvar(r2->argument());
		auto dep2 = delta->add_ctxvar(dep);
		delta1 = delta->finalize(create_testop(delta->subregion(), {dep1, dep2}, {&vt})[0]);
	}

	{
		auto delta = delta::node::create(
			region,
			vt,
			"test-delta2",
			linkage::external_linkage,
			false);
		auto dep1 = delta->add_ctxvar(r1->argument());
		auto dep2 = delta->add_ctxvar(dep);
		delta2 = delta->finalize(create_testop(delta->subregion(), {dep1, dep2}, {&vt})[0]);
	}

	r1->set_rvorigin(delta1);
	r2->set_rvorigin(delta2);

	auto phi = pb.end();
	rm.graph()->add_export(phi->output(0), {phi->output(0)->type(), ""});

	jive::view(*rm.graph(), stdout);

	StatisticsDescriptor sd;
	auto module = rvsdg2jlm::rvsdg2jlm(rm, sd);
	jlm::print(*module, stdout);

	/* verify output */

	auto & ipg = module->ipgraph();
	assert(ipg.nnodes() == 3);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/backend/llvm/r2j/test-recursive-data", test)
