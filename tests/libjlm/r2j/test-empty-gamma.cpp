/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/rvsdg/control.h>
#include <jive/rvsdg/gamma.h>
#include <jive/types/function/fcttype.h>
#include <jive/view.h>

#include <jlm/ir/lambda.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg.hpp>
#include <jlm/ir/view.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>

static void
test_with_match()
{
	jlm::valuetype vt;
	jive::bittype bt1(1);
	jive::fct::type ft({&bt1, &vt, &vt}, {&vt});

	jlm::rvsdg rvsdg("", "");
	auto nf = rvsdg.graph()->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	/* setup graph */

	jlm::lambda_builder lb;
	auto arguments = lb.begin_lambda(rvsdg.graph()->root(), ft);

	auto match = jive::match(1, {{0, 0}}, 1, 2, arguments[0]);
	auto gamma = jive::gamma_node::create(match, 2);
	auto ev1 = gamma->add_entryvar(arguments[1]);
	auto ev2 = gamma->add_entryvar(arguments[2]);
	auto ex = gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});

	auto lambda = lb.end_lambda({ex});
	rvsdg.graph()->add_export(lambda->output(0), "");


	jive::view(*rvsdg.graph(), stdout);
	auto module = jlm::rvsdg2jlm::rvsdg2jlm(rvsdg);
	jlm::view(*module, stdout);

	/* verify output */

	auto & clg = module->callgraph();
	assert(clg.nnodes() == 1);

	auto cfg = dynamic_cast<const jlm::function_node&>(*clg.begin()).cfg();
	assert(cfg->nnodes() == 3);
	auto node = cfg->entry_node()->outedge(0)->sink();
	auto bb = dynamic_cast<const jlm::basic_block*>(&node->attribute());
	assert(jlm::is_select_op(bb->last()->operation()));
}

static void
test_without_match()
{
	jlm::valuetype vt;
	jive::ctltype ctl2(2);
	jive::bittype bt1(1);
	jive::fct::type ft({&ctl2, &vt, &vt}, {&vt});

	jlm::rvsdg rvsdg("", "");
	auto nf = rvsdg.graph()->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	/* setup graph */

	jlm::lambda_builder lb;
	auto arguments = lb.begin_lambda(rvsdg.graph()->root(), ft);

	auto gamma = jive::gamma_node::create(arguments[0], 2);
	auto ev1 = gamma->add_entryvar(arguments[1]);
	auto ev2 = gamma->add_entryvar(arguments[2]);
	auto ex = gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});

	auto lambda = lb.end_lambda({ex});
	rvsdg.graph()->add_export(lambda->output(0), "");


	jive::view(*rvsdg.graph(), stdout);
	auto module = jlm::rvsdg2jlm::rvsdg2jlm(rvsdg);
	jlm::view(*module, stdout);

	/* verify output */

	auto & clg = module->callgraph();
	assert(clg.nnodes() == 1);

	auto cfg = dynamic_cast<const jlm::function_node&>(*clg.begin()).cfg();
	assert(cfg->nnodes() == 3);
	auto node = cfg->entry_node()->outedge(0)->sink();
	auto bb = dynamic_cast<const jlm::basic_block*>(&node->attribute());
	assert(jlm::is_ctl2bits_op(bb->first()->operation()));
	assert(jlm::is_select_op(bb->last()->operation()));
}

static int
test()
{
	test_with_match();
	test_without_match();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/r2j/test-empty-gamma", test)
