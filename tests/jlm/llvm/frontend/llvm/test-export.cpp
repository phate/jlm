/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
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
	FunctionType ft({&vt}, {&vt});

	ipgraph_module im(filepath(""), "", "");

	auto d = data_node::Create(
    im.ipgraph(),
    "d",
    vt,
    linkage::external_linkage,
    "",
    false);
	auto f = function_node::create(im.ipgraph(), "f", ft, linkage::external_linkage);

	im.create_global_value(d);
	im.create_variable(f);

	StatisticsCollector statisticsCollector;
	auto rvsdgModule = ConvertInterProceduralGraphModule(im, statisticsCollector);

	jive::view(rvsdgModule->Rvsdg(), stdout);

	/*
		We should have no exports in the RVSDG. The data and function
		node should be converted to RVSDG imports as they do not have
		a body, i.e., either a CFG or a initialization.
	*/
	assert(rvsdgModule->Rvsdg().root()->nresults() == 0);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-export", test)
