/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/types/function.hpp>
#include <jive/view.hpp>

#include <jlm/frontend/llvm/jlm2rvsdg/module.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/util/Statistics.hpp>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	jive::fcttype ft({&vt}, {&vt});

	ipgraph_module im(filepath(""), "", "");

	auto d = data_node::create(im.ipgraph(), "d", vt, linkage::external_linkage, false);
	auto f = function_node::create(im.ipgraph(), "f", ft, linkage::external_linkage);

	im.create_global_value(d);
	im.create_variable(f);

	StatisticsDescriptor sd;
	auto rvsdg = construct_rvsdg(im, sd);

	jive::view(*rvsdg->graph(), stdout);

	/*
		We should have no exports in the RVSDG. The data and function
		node should be converted to RVSDG imports as they do not have
		a body, i.e., either a CFG or a initialization.
	*/
	assert(rvsdg->graph()->root()->nresults() == 0);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/frontend/llvm/jlm-rvsdg/test-export", test)
