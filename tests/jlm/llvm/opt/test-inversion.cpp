/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inversion.hpp>
#include <jlm/util/Statistics.hpp>

static const jlm::valuetype vt;
static jlm::StatisticsCollector statisticsCollector;

static inline void
test1()
{
	using namespace jlm;

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();

	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});
	auto z = graph.add_import({vt, "z"});

	auto theta = jive::theta_node::create(graph.root());

	auto lvx = theta->add_loopvar(x);
	auto lvy = theta->add_loopvar(y);
	theta->add_loopvar(z);

	auto a = jlm::create_testop(theta->subregion(), {lvx->argument(), lvy->argument()},
		{&jive::bit1})[0];
	auto predicate = jive::match(1, {{1, 0}}, 1, 2, a);

	auto gamma = jive::gamma_node::create(predicate, 2);

	auto evx = gamma->add_entryvar(lvx->argument());
	auto evy = gamma->add_entryvar(lvy->argument());

	auto b = jlm::create_testop(gamma->subregion(0), {evx->argument(0), evy->argument(0)}, {&vt})[0];
	auto c = jlm::create_testop(gamma->subregion(1), {evx->argument(1), evy->argument(1)}, {&vt})[0];

	auto xvy = gamma->add_exitvar({b, c});

	lvy->result()->divert_to(xvy);

	theta->set_predicate(predicate);

	auto ex1 = graph.add_export(theta->output(0), {theta->output(0)->type(), "x"});
	auto ex2 = graph.add_export(theta->output(1), {theta->output(1)->type(), "y"});
	auto ex3 = graph.add_export(theta->output(2), {theta->output(2)->type(), "z"});

//	jive::view(graph.root(), stdout);
	jlm::tginversion tginversion;
	tginversion.run(rm, statisticsCollector);
//	jive::view(graph.root(), stdout);

	assert(jive::is<jive::gamma_op>(jive::node_output::node(ex1->origin())));
	assert(jive::is<jive::gamma_op>(jive::node_output::node(ex2->origin())));
	assert(jive::is<jive::gamma_op>(jive::node_output::node(ex3->origin())));
}

static inline void
test2()
{
	using namespace jlm;

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();

	auto x = graph.add_import({vt, "x"});

	auto theta = jive::theta_node::create(graph.root());

	auto lv1 = theta->add_loopvar(x);

	auto n1 = jlm::create_testop(theta->subregion(), {lv1->argument()}, {&jive::bit1})[0];
	auto n2 = jlm::create_testop(theta->subregion(), {lv1->argument()}, {&vt})[0];
	auto predicate = jive::match(1, {{1, 0}}, 1, 2, n1);

	auto gamma = jive::gamma_node::create(predicate, 2);

	auto ev1 = gamma->add_entryvar(n1);
	auto ev2 = gamma->add_entryvar(lv1->argument());
	auto ev3 = gamma->add_entryvar(n2);

	gamma->add_exitvar({ev1->argument(0), ev1->argument(1)});
	gamma->add_exitvar({ev2->argument(0), ev2->argument(1)});
	gamma->add_exitvar({ev3->argument(0), ev3->argument(1)});

	lv1->result()->divert_to(gamma->output(1));

	theta->set_predicate(predicate);

	auto ex = graph.add_export(theta->output(0), {theta->output(0)->type(), "x"});

//	jive::view(graph.root(), stdout);
	jlm::tginversion tginversion;
	tginversion.run(rm, statisticsCollector);
//	jive::view(graph.root(), stdout);

	assert(jive::is<jive::gamma_op>(jive::node_output::node(ex->origin())));
}

static int
verify()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inversion", verify)
