/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/view.hpp>
#include <jive/rvsdg/control.hpp>
#include <jive/rvsdg/gamma.hpp>

#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/inlining.hpp>
#include <jlm/util/stats.hpp>

static const jlm::stats_descriptor sd;

static void
test1()
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
	auto lambda1 = lambda::node::create(graph.root(), ft1, "f1", linkage::external_linkage);
	lambda1->add_ctxvar(i);
	auto t = test_op::create(lambda1->subregion(), {lambda1->fctargument(0)}, {&vt});
	auto f1 = lambda1->finalize({t->output(0)});

	/* f2 */
	auto lambda2 = lambda::node::create(graph.root(), ft2, "f1", linkage::external_linkage);
	auto d = lambda2->add_ctxvar(f1);

	auto gamma = jive::gamma_node::create(lambda2->fctargument(0), 2);
	auto ev1 = gamma->add_entryvar(lambda2->fctargument(1));
	auto ev2 = gamma->add_entryvar(d);
	auto apply = call_op::create(ev2->argument(0), {ev1->argument(0)})[0];
	auto xv1 = gamma->add_exitvar({apply, ev1->argument(1)});
	auto f2 = lambda2->finalize({xv1});

	graph.add_export(f2, {f2->type(), "f2"});

//	jive::view(graph.root(), stdout);
	jlm::fctinline fctinline;
	fctinline.run(rm, sd);
//	jive::view(graph.root(), stdout);

	assert(!jive::contains<jlm::call_op>(graph.root(), true));
}

static void
test2()
{
	using namespace jlm;

	valuetype vt;
	statetype st;
	jive::fcttype ft1({&vt, &st}, {&st});
	jive::fcttype ft2({&st}, {&st});
	ptrtype pt(ft1);
	jive::fcttype ft3({&pt, &st}, {&st});

	rvsdg_module rm(filepath(""), "", "");
	auto & graph = *rm.graph();
	auto i = graph.add_import({ptrtype(ft3), "i"});

	/* f1 */
	auto f1Lambda = lambda::node::create(graph.root(), ft1, "f1", linkage::external_linkage);
	auto f1 = f1Lambda->finalize({f1Lambda->fctargument(1)});

	/* f2 */
	auto f2Lambda = lambda::node::create(graph.root(), ft2, "f2", linkage::external_linkage);
	auto cvi = f2Lambda->add_ctxvar(i);
	auto cvf1 = f2Lambda->add_ctxvar(f1);

	auto call = call_op::create(cvi, {cvf1, f2Lambda->fctargument(0)});

	auto f2 = f2Lambda->finalize({call[0]});

	graph.add_export(f2, {f2->type(), "f2"});

	jive::view(graph.root(), stdout);
	jlm::fctinline fctinline;
	fctinline.run(rm, sd);
	jive::view(graph.root(), stdout);

	/*
		Function f1 should not have been inlined.
	*/
	assert(is<call_op>(jive::node_output::node(f2Lambda->fctresult(0)->origin())));
}

static int
verify()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-inlining", verify)
