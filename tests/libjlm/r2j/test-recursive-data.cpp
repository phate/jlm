/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <jive/rvsdg/phi.h>
#include <jive/view.h>

#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators/delta.hpp>
#include <jlm/ir/print.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/util/stats.hpp>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	ptrtype pt(vt);

	rvsdg_module rm(filepath(""), "", "");

	/* setup graph */
	auto imp = rm.graph()->add_import({pt, ""});

	jive::phi_builder pb;
	auto region = pb.begin_phi(rm.graph()->root());
	auto r1 = pb.add_recvar(pt);
	auto r2 = pb.add_recvar(pt);
	auto dep = pb.add_dependency(imp);

	jive::output * delta1, * delta2;
	{
		delta_builder db;
		auto subregion = db.begin(region, vt, "test-delta1", linkage::external_linkage, false);
		auto dep1 = db.add_dependency(r2->value());
		auto dep2 = db.add_dependency(dep);
		delta1 = db.end(create_testop(subregion, {dep1, dep2}, {&vt})[0]);
	}

	{
		delta_builder db;
		auto subregion = db.begin(region, vt, "test-delta2", linkage::external_linkage, false);
		auto dep1 = db.add_dependency(r1->value());
		auto dep2 = db.add_dependency(dep);
		delta2 = db.end(create_testop(subregion, {dep1, dep2}, {&vt})[0]);
	}

	r1->set_value(delta1);
	r2->set_value(delta2);

	auto phi = pb.end_phi();
	rm.graph()->add_export(phi->output(0), {phi->output(0)->type(), ""});

	jive::view(*rm.graph(), stdout);

	stats_descriptor sd;
	auto module = rvsdg2jlm::rvsdg2jlm(rm, sd);
	jlm::print(*module, stdout);

	/* verify output */

	auto & ipg = module->ipgraph();
	assert(ipg.nnodes() == 3);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/r2j/test-recursive-data", test)
