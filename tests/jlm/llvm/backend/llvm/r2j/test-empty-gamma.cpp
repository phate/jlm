/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

static void
test_with_match()
{
	using namespace jlm::llvm;

	jlm::valuetype vt;
	jlm::rvsdg::bittype bt1(1);
	FunctionType ft({&bt1, &vt, &vt}, {&vt});

	RvsdgModule rm(jlm::util::filepath(""), "", "");
	auto nf = rm.Rvsdg().node_normal_form(typeid(jlm::rvsdg::operation));
	nf->set_mutable(false);

	/* setup graph */

	auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

	auto match = jlm::rvsdg::match(1, {{0, 0}}, 1, 2, lambda->fctargument(0));
	auto gamma = jlm::rvsdg::gamma_node::create(match, 2);
	auto ev1 = gamma->add_entryvar(lambda->fctargument(1));
	auto ev2 = gamma->add_entryvar(lambda->fctargument(2));
	auto ex = gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});

	auto f = lambda->finalize({ex});
  rm.Rvsdg().add_export(f, {f->type(), ""});

	jlm::rvsdg::view(rm.Rvsdg(), stdout);

	jlm::util::StatisticsCollector statisticsCollector;
	auto module = rvsdg2jlm::rvsdg2jlm(rm, statisticsCollector);
	print(*module, stdout);

	/* verify output */

	auto & ipg = module->ipgraph();
	assert(ipg.nnodes() == 1);

	auto cfg = dynamic_cast<const function_node&>(*ipg.begin()).cfg();
	assert(cfg->nnodes() == 1);
	auto node = cfg->entry()->outedge(0)->sink();
	auto bb = dynamic_cast<const basic_block*>(node);
	assert(jlm::rvsdg::is<select_op>(bb->tacs().last()->operation()));
}

static void
test_without_match()
{
	using namespace jlm::llvm;

	jlm::valuetype vt;
	jlm::rvsdg::ctltype ctl2(2);
	jlm::rvsdg::bittype bt1(1);
	FunctionType ft({&ctl2, &vt, &vt}, {&vt});

	RvsdgModule rm(jlm::util::filepath(""), "", "");
	auto nf = rm.Rvsdg().node_normal_form(typeid(jlm::rvsdg::operation));
	nf->set_mutable(false);

	/* setup graph */

	auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

	auto gamma = jlm::rvsdg::gamma_node::create(lambda->fctargument(0), 2);
	auto ev1 = gamma->add_entryvar(lambda->fctargument(1));
	auto ev2 = gamma->add_entryvar(lambda->fctargument(2));
	auto ex = gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});

	auto f = lambda->finalize({ex});
  rm.Rvsdg().add_export(f, {f->type(), ""});

	jlm::rvsdg::view(rm.Rvsdg(), stdout);

	jlm::util::StatisticsCollector statisticsCollector;
	auto module = rvsdg2jlm::rvsdg2jlm(rm, statisticsCollector);
	print(*module, stdout);

	/* verify output */

	auto & ipg = module->ipgraph();
	assert(ipg.nnodes() == 1);

	auto cfg = dynamic_cast<const function_node&>(*ipg.begin()).cfg();
	assert(cfg->nnodes() == 1);
	auto node = cfg->entry()->outedge(0)->sink();
	auto bb = dynamic_cast<const basic_block*>(node);
	assert(jlm::rvsdg::is<ctl2bits_op>(bb->tacs().first()->operation()));
	assert(jlm::rvsdg::is<select_op>(bb->tacs().last()->operation()));
}

static void
test_gamma3()
{
	using namespace jlm::llvm;

	jlm::valuetype vt;
	FunctionType ft({&jlm::rvsdg::bit32, &vt, &vt}, {&vt});

	RvsdgModule rm(jlm::util::filepath(""), "", "");
	auto nf = rm.Rvsdg().node_normal_form(typeid(jlm::rvsdg::operation));
	nf->set_mutable(false);

	/* setup graph */

	auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

	auto match = jlm::rvsdg::match(32, {{0, 0}, {1, 1}}, 2, 3, lambda->fctargument(0));

	auto gamma = jlm::rvsdg::gamma_node::create(match, 3);
	auto ev1 = gamma->add_entryvar(lambda->fctargument(1));
	auto ev2 = gamma->add_entryvar(lambda->fctargument(2));
	auto ex = gamma->add_exitvar({ev1->argument(0), ev1->argument(1), ev2->argument(2)});

	auto f = lambda->finalize({ex});
  rm.Rvsdg().add_export(f, {f->type(), ""});

	jlm::rvsdg::view(rm.Rvsdg(), stdout);

	jlm::util::StatisticsCollector statisticsCollector;
	auto module = rvsdg2jlm::rvsdg2jlm(rm, statisticsCollector);
	print(*module, stdout);

	/* verify output */

	auto & ipg = module->ipgraph();
	assert(ipg.nnodes() == 1);

	auto cfg = dynamic_cast<const function_node&>(*ipg.begin()).cfg();
	assert(is_closed(*cfg));
}

static int
test()
{
	test_with_match();
	test_without_match();

	test_gamma3();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/r2j/test-empty-gamma", test)
