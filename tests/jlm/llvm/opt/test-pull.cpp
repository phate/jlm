/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>
#include <jlm/rvsdg/gamma.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/util/Statistics.hpp>

static const jlm::valuetype vt;
static jlm::StatisticsCollector statisticsCollector;

static inline void
test_pullin_top()
{
	using namespace jlm;

	jive::ctltype ct(2);
	jlm::test_op uop({&vt}, {&vt});
	jlm::test_op bop({&vt, &vt}, {&vt});
	jlm::test_op cop({&ct, &vt}, {&ct});

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();

	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});

	auto n1 = jlm::create_testop(graph.root(), {x}, {&vt})[0];
	auto n2 = jlm::create_testop(graph.root(), {x}, {&vt})[0];
	auto n3 = jlm::create_testop(graph.root(), {n2}, {&vt})[0];
	auto n4 = jlm::create_testop(graph.root(), {c, n1}, {&ct})[0];
	auto n5 = jlm::create_testop(graph.root(), {n1, n3}, {&vt})[0];

	auto gamma = jive::gamma_node::create(n4, 2);

	gamma->add_entryvar(n4);
	auto ev = gamma->add_entryvar(n5);
	gamma->add_exitvar({ev->argument(0), ev->argument(1)});

	graph.add_export(gamma->output(0), {gamma->output(0)->type(), "x"});
	graph.add_export(n2, {n2->type(), "y"});

//	jive::view(graph, stdout);
	jlm::pullin_top(gamma);
//	jive::view(graph, stdout);

	assert(gamma->subregion(0)->nnodes() == 2);
	assert(gamma->subregion(1)->nnodes() == 2);
}

static inline void
test_pullin_bottom()
{
	jlm::valuetype vt;
	jive::ctltype ct(2);

	jive::graph graph;
	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});

	auto gamma = jive::gamma_node::create(c, 2);

	auto ev = gamma->add_entryvar(x);
	gamma->add_exitvar({ev->argument(0), ev->argument(1)});

	auto b1 = jlm::create_testop(graph.root(), {gamma->output(0), x}, {&vt})[0];
	auto b2 = jlm::create_testop(graph.root(), {gamma->output(0), b1}, {&vt})[0];

	auto xp = graph.add_export(b2, {b2->type(), "x"});

//	jive::view(graph, stdout);
	jlm::pullin_bottom(gamma);
//	jive::view(graph, stdout);

	assert(jive::node_output::node(xp->origin()) == gamma);
	assert(gamma->subregion(0)->nnodes() == 2);
	assert(gamma->subregion(1)->nnodes() == 2);
}

static void
test_pull()
{
	using namespace jlm;

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();

	auto p = graph.add_import({jive::ctl2, ""});

	auto croot = jlm::create_testop(graph.root(), {}, {&vt})[0];

	/* outer gamma */
	auto gamma1 = jive::gamma_node::create(p, 2);
	auto ev1 = gamma1->add_entryvar(p);
	auto ev2 = gamma1->add_entryvar(croot);

	auto cg1 = jlm::create_testop(gamma1->subregion(0), {}, {&vt})[0];

	/* inner gamma */
	auto gamma2 = jive::gamma_node::create(ev1->argument(1), 2);
	auto ev3 = gamma2->add_entryvar(ev2->argument(1));
	auto cg2 = jlm::create_testop(gamma2->subregion(0), {}, {&vt})[0];
	auto un = jlm::create_testop(gamma2->subregion(1), {ev3->argument(1)}, {&vt})[0];
	auto g2xv = gamma2->add_exitvar({cg2, un});

	auto g1xv = gamma1->add_exitvar({cg1, g2xv});

	graph.add_export(g1xv, {g1xv->type(), ""});

	jive::view(graph, stdout);
	jlm::pullin pullin;
	pullin.run(rm, statisticsCollector);
	graph.prune();
	jive::view(graph, stdout);

	assert(graph.root()->nnodes() == 1);
}

static int
verify()
{
	test_pullin_top();
	test_pullin_bottom();

	test_pull();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-pull", verify)
