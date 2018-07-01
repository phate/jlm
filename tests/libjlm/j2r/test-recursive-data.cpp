/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <jive/view.h>

#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/view.hpp>
#include <jlm/jlm/ir/rvsdg.hpp>
#include <jlm/jlm/jlm2rvsdg/module.hpp>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	module m("", "");

	auto d0 = data_node::create(m.ipgraph(), "d0", vt, linkage::external_linkage, false);

	auto d1 = data_node::create(m.ipgraph(), "d1", vt, linkage::external_linkage, false);
	auto d2 = data_node::create(m.ipgraph(), "d2", vt, linkage::external_linkage, false);

	auto v0 = m.create_global_value(d0);
	auto v1 = m.create_global_value(d1);
	auto v2 = m.create_global_value(d2);

	d1->add_dependency(d0);
	d1->add_dependency(d2);
	d2->add_dependency(d0);
	d2->add_dependency(d1);

	tacsvector_t tvec1, tvec2;
	auto tv1 = m.create_tacvariable(vt);
	auto tv2 = m.create_tacvariable(vt);
	tvec1.push_back(std::move(create_testop_tac({v0, v2}, {tv1})));
	tvec2.push_back(std::move(create_testop_tac({v0, v1}, {tv2})));

	d1->set_initialization(std::move(tvec1));
	d2->set_initialization(std::move(tvec2));

	auto rvsdg = construct_rvsdg(m);

	jive::view(*rvsdg->graph(), stdout);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/j2r/test-recursive-data", test)
