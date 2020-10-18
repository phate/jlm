/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/rvsdg/control.hpp>
#include <jive/rvsdg/gamma.hpp>
#include <jive/view.hpp>

#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators/lambda.hpp>
#include <jlm/ir/print.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/util/stats.hpp>

static int
test()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::bittype bt1(1);
	jive::fcttype ft({&bt1, &vt}, {&vt});

	rvsdg_module rm(filepath(""), "", "");

	jlm::lambda_builder lb;
	auto arguments = lb.begin_lambda(rm.graph()->root(), {ft, "f", linkage::external_linkage});

	auto match = jive::match(1, {{0, 0}}, 1, 2, arguments[0]);
	auto gamma = jive::gamma_node::create(match, 2);
	auto ev = gamma->add_entryvar(arguments[1]);
	auto output = jlm::create_testop(gamma->subregion(1), {ev->argument(1)}, {&vt})[0];
	auto ex = gamma->add_exitvar({ev->argument(0), output});

	auto lambda = lb.end_lambda({ex});

	rm.graph()->add_export(lambda->output(0), {lambda->output(0)->type(), ""});

	jive::view(*rm.graph(), stdout);

	stats_descriptor sd;
	auto module = rvsdg2jlm::rvsdg2jlm(rm, sd);
	auto & ipg = module->ipgraph();
	assert(ipg.nnodes() == 1);

	auto cfg = dynamic_cast<const jlm::function_node&>(*ipg.begin()).cfg();
	jlm::print_ascii(*cfg, stdout);

	assert(!is_proper_structured(*cfg));
	assert(is_structured(*cfg));

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/r2j/test-partial-gamma", test)
