/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/view.h>
#include <jive/rvsdg/control.h>
#include <jive/rvsdg/gamma.h>

#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/inlining.hpp>
#include <jlm/util/stats.hpp>

static const jlm::stats_descriptor sd;

static int
verify()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);
	jive::fcttype ft1({&vt}, {&vt});
	jive::fcttype ft2({&ct, &vt}, {&vt});

	rvsdg_module rm(filepath(""), "", "");
	auto & graph = *rm.graph();
	auto i = graph.add_import({vt, "i"});

	/* f1 */
	jlm::lambda_builder lb;
	auto arguments = lb.begin_lambda(graph.root(), {ft1, "f1", linkage::external_linkage});
	lb.add_dependency(i);
	auto t = jlm::create_testop(lb.subregion(), {arguments[0]}, {&vt})[0]->node();
	auto f1 = lb.end_lambda({t->output(0)});

	/* f2 */
	arguments = lb.begin_lambda(graph.root(), {ft2, "f2", linkage::external_linkage});
	auto d = lb.add_dependency(f1->output(0));

	auto gamma = jive::gamma_node::create(arguments[0], 2);
	auto ev1 = gamma->add_entryvar(arguments[1]);
	auto ev2 = gamma->add_entryvar(d);
	auto apply = call_op::create(ev2->argument(0), {ev1->argument(0)})[0];
	auto xv1 = gamma->add_exitvar({apply, ev1->argument(1)});
	auto f2 = lb.end_lambda({xv1});

	graph.add_export(f2->output(0), {f2->output(0)->type(), "f2"});

	jive::view(graph.root(), stdout);
	jlm::fctinline fctinline;
	fctinline.run(rm, sd);
	jive::view(graph.root(), stdout);

	assert(!jive::contains<jlm::call_op>(graph.root(), true));
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-inlining", verify)
