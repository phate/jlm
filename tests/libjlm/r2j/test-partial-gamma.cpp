/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/rvsdg/control.h>
#include <jive/rvsdg/gamma.h>
#include <jive/view.h>

#include <jlm/jlm/ir/cfg-structure.hpp>
#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/operators/lambda.hpp>
#include <jlm/jlm/ir/rvsdg.hpp>
#include <jlm/jlm/ir/view.hpp>
#include <jlm/jlm/rvsdg2jlm/rvsdg2jlm.hpp>

static int
test()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::bittype bt1(1);
	jive::fcttype ft({&bt1, &vt}, {&vt});

	jlm::rvsdg rvsdg("", "");

	jlm::lambda_builder lb;
	auto arguments = lb.begin_lambda(rvsdg.graph()->root(), {ft, "f", linkage::external_linkage});

	auto match = jive::match(1, {{0, 0}}, 1, 2, arguments[0]);
	auto gamma = jive::gamma_node::create(match, 2);
	auto ev = gamma->add_entryvar(arguments[1]);
	auto output = jlm::create_testop(gamma->subregion(1), {ev->argument(1)}, {&vt})[0];
	auto ex = gamma->add_exitvar({ev->argument(0), output});

	auto lambda = lb.end_lambda({ex});

	rvsdg.graph()->add_export(lambda->output(0), "");

	jive::view(*rvsdg.graph(), stdout);

	auto module = jlm::rvsdg2jlm::rvsdg2jlm(rvsdg);
	auto & ipg = module->ipgraph();
	assert(ipg.nnodes() == 1);

	auto cfg = dynamic_cast<const jlm::function_node&>(*ipg.begin()).cfg();
	jlm::view_ascii(*cfg, stdout);

	assert(!is_proper_structured(*cfg));
	assert(is_structured(*cfg));

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/r2j/test-partial-gamma", test)
