/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	ipgraph_module im(filepath(""), "", "");

	auto d0 = data_node::Create(
    im.ipgraph(),
    "d0",
    vt,
    linkage::external_linkage,
    "",
    false);

	auto d1 = data_node::Create(
    im.ipgraph(),
    "d1",
    vt,
    linkage::external_linkage,
    "",
    false);
	auto d2 = data_node::Create(
    im.ipgraph(),
    "d2",
    vt,
    linkage::external_linkage,
    "",
    false);

	auto v0 = im.create_global_value(d0);
	auto v1 = im.create_global_value(d1);
	auto v2 = im.create_global_value(d2);

	d1->add_dependency(d0);
	d1->add_dependency(d2);
	d2->add_dependency(d0);
	d2->add_dependency(d1);

	tacsvector_t tvec1, tvec2;
	tvec1.push_back(create_testop_tac({v0, v2}, {&vt}));
	tvec2.push_back(create_testop_tac({v0, v1}, {&vt}));

	d1->set_initialization(std::make_unique<data_node_init>(std::move(tvec1)));
	d2->set_initialization(std::make_unique<data_node_init>(std::move(tvec2)));

	StatisticsCollector statisticsCollector;
	auto rvsdgModule = ConvertInterProceduralGraphModule(im, statisticsCollector);

	jive::view(rvsdgModule->Rvsdg(), stdout);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-recursive-data", test)
