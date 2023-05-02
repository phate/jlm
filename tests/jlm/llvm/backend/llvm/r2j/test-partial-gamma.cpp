/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

static int
test()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::bittype bt1(1);
	FunctionType ft({&bt1, &vt}, {&vt});

	RvsdgModule rm(filepath(""), "", "");

	auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

	auto match = jive::match(1, {{0, 0}}, 1, 2, lambda->fctargument(0));
	auto gamma = jive::gamma_node::create(match, 2);
	auto ev = gamma->add_entryvar(lambda->fctargument(1));
	auto output = jlm::create_testop(gamma->subregion(1), {ev->argument(1)}, {&vt})[0];
	auto ex = gamma->add_exitvar({ev->argument(0), output});

	auto f = lambda->finalize({ex});

  rm.Rvsdg().add_export(f, {f->type(), ""});

	jive::view(rm.Rvsdg(), stdout);

	StatisticsCollector statisticsCollector;
	auto module = rvsdg2jlm::rvsdg2jlm(rm, statisticsCollector);
	auto & ipg = module->ipgraph();
	assert(ipg.nnodes() == 1);

	auto cfg = dynamic_cast<const jlm::function_node&>(*ipg.begin()).cfg();
	jlm::print_ascii(*cfg, stdout);

	assert(!is_proper_structured(*cfg));
	assert(is_structured(*cfg));

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/r2j/test-partial-gamma", test)
