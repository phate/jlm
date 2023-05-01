/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Phi.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunDeadNodeElimination(jlm::RvsdgModule & rvsdgModule)
{
  jlm::StatisticsCollector statisticsCollector;
  jlm::DeadNodeElimination deadNodeElimination;
  deadNodeElimination.run(rvsdgModule, statisticsCollector);
}

static void
TestRoot()
{
	using namespace jlm;

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	graph.add_import({jlm::valuetype(), "x"});
	auto y = graph.add_import({jlm::valuetype(), "y"});
	graph.add_export(y, {y->type(), "z"});

//	jive::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
//	jive::view(graph.root(), stdout);

	assert(graph.root()->narguments() == 1);
}

static void
TestGamma()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});

	auto gamma = jive::gamma_node::create(c, 2);
	auto ev1 = gamma->add_entryvar(x);
	auto ev2 = gamma->add_entryvar(y);
	auto ev3 = gamma->add_entryvar(x);

	auto t = jlm::create_testop(gamma->subregion(1), {ev2->argument(1)}, {&vt})[0];

	gamma->add_exitvar({ev1->argument(0), ev1->argument(1)});
	gamma->add_exitvar({ev2->argument(0), t});
	gamma->add_exitvar({ev3->argument(0), ev1->argument(1)});

	graph.add_export(gamma->output(0), {gamma->output(0)->type(), "z"});
	graph.add_export(gamma->output(2), {gamma->output(2)->type(), "w"});

//	jive::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
//	jive::view(graph.root(), stdout);

	assert(gamma->noutputs() == 2);
	assert(gamma->subregion(1)->nodes.empty());
	assert(gamma->subregion(1)->narguments() == 2);
	assert(gamma->ninputs() == 3);
	assert(graph.root()->narguments() == 2);
}

static void
TestGamma2()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});

	auto gamma = jive::gamma_node::create(c, 2);
	gamma->add_entryvar(x);

	auto n1 = jlm::create_testop(gamma->subregion(0), {}, {&vt})[0];
	auto n2 = jlm::create_testop(gamma->subregion(1), {}, {&vt})[0];

	gamma->add_exitvar({n1, n2});

	graph.add_export(gamma->output(0), {gamma->output(0)->type(), "x"});

//	jive::view(graph, stdout);
  RunDeadNodeElimination(rm);
//	jive::view(graph, stdout);

	assert(graph.root()->narguments() == 1);
}

static void
TestTheta()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});
	auto z = graph.add_import({vt, "z"});

	auto theta = jive::theta_node::create(graph.root());

	auto lv1 = theta->add_loopvar(x);
	auto lv2 = theta->add_loopvar(y);
	auto lv3 = theta->add_loopvar(z);
	auto lv4 = theta->add_loopvar(y);

	lv1->result()->divert_to(lv2->argument());
	lv2->result()->divert_to(lv1->argument());

	auto t = jlm::create_testop(theta->subregion(), {lv3->argument()}, {&vt})[0];
	lv3->result()->divert_to(t);
	lv4->result()->divert_to(lv2->argument());

	auto c = jlm::create_testop(theta->subregion(), {}, {&ct})[0];
	theta->set_predicate(c);

	graph.add_export(theta->output(0), {theta->output(0)->type(), "a"});
	graph.add_export(theta->output(3), {theta->output(0)->type(), "b"});

//	jive::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
//	jive::view(graph.root(), stdout);

	assert(theta->noutputs() == 3);
	assert(theta->subregion()->nodes.size() == 1);
	assert(graph.root()->narguments() == 2);
}

static void
TestNestedTheta()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto c = graph.add_import({ct, "c"});
	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});

	auto otheta = jive::theta_node::create(graph.root());

	auto lvo1 = otheta->add_loopvar(c);
	auto lvo2 = otheta->add_loopvar(x);
	auto lvo3 = otheta->add_loopvar(y);

	auto itheta = jive::theta_node::create(otheta->subregion());

	auto lvi1 = itheta->add_loopvar(lvo1->argument());
	auto lvi2 = itheta->add_loopvar(lvo2->argument());
	auto lvi3 = itheta->add_loopvar(lvo3->argument());

	lvi2->result()->divert_to(lvi3->argument());

	itheta->set_predicate(lvi1->argument());

	lvo2->result()->divert_to(itheta->output(1));
	lvo3->result()->divert_to(itheta->output(1));

	otheta->set_predicate(lvo1->argument());

	graph.add_export(otheta->output(2), {otheta->output(2)->type(), "y"});

//	jive::view(graph, stdout);
  RunDeadNodeElimination(rm);
//	jive::view(graph, stdout);

	assert(otheta->noutputs() == 3);
}

static void
TestEvolvingTheta()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ct(2);

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto c = graph.add_import({ct, "c"});
	auto x1 = graph.add_import({vt, "x1"});
	auto x2 = graph.add_import({vt, "x2"});
	auto x3 = graph.add_import({vt, "x3"});
	auto x4 = graph.add_import({vt, "x4"});

	auto theta = jive::theta_node::create(graph.root());

	auto lv0 = theta->add_loopvar(c);
	auto lv1 = theta->add_loopvar(x1);
	auto lv2 = theta->add_loopvar(x2);
	auto lv3 = theta->add_loopvar(x3);
	auto lv4 = theta->add_loopvar(x4);

	lv1->result()->divert_to(lv2->argument());
	lv2->result()->divert_to(lv3->argument());
	lv3->result()->divert_to(lv4->argument());

	theta->set_predicate(lv0->argument());

	graph.add_export(lv1, {lv1->type(), "x1"});

//	jive::view(graph, stdout);
  RunDeadNodeElimination(rm);
//	jive::view(graph, stdout);

	assert(theta->noutputs() == 5);
}

static void
TestLambda()
{
	using namespace jlm;

	jlm::valuetype vt;

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});

	auto lambda = lambda::node::create(graph.root(), {{&vt}, {&vt, &vt}}, "f",
		linkage::external_linkage);

	auto cv1 = lambda->add_ctxvar(x);
	auto cv2 = lambda->add_ctxvar(y);
	jlm::create_testop(lambda->subregion(), {lambda->fctargument(0), cv1}, {&vt});

	auto output = lambda->finalize({lambda->fctargument(0), cv2});

	graph.add_export(output, {output->type(), "f"});

//	jive::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
//	jive::view(graph.root(), stdout);

	assert(lambda->subregion()->nodes.empty());
	assert(graph.root()->narguments() == 1);
}

static void
TestPhi()
{
	using namespace jlm;

	jlm::valuetype vt;
	FunctionType ft({&vt}, {&vt});

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto x = graph.add_import({vt, "x"});
	auto y = graph.add_import({vt, "y"});

	phi::builder pb;
	pb.begin(graph.root());

	auto rv1 = pb.add_recvar(PointerType());
	auto rv2 = pb.add_recvar(PointerType());
	auto dx = pb.add_ctxvar(x);
	auto dy = pb.add_ctxvar(y);

	auto lambda1 = lambda::node::create(pb.subregion(), ft, "f", linkage::external_linkage);
	lambda1->add_ctxvar(rv1->argument());
	lambda1->add_ctxvar(dx);
	auto f1 = lambda1->finalize({lambda1->fctargument(0)});

	auto lambda2 = lambda::node::create(pb.subregion(), ft, "f", linkage::external_linkage);
	lambda2->add_ctxvar(rv2->argument());
	lambda2->add_ctxvar(dy);
	auto f2 = lambda2->finalize({lambda2->fctargument(0)});

	rv1->set_rvorigin(f1);
	rv2->set_rvorigin(f2);
	auto phi = pb.end();

	graph.add_export(phi->output(0), {phi->output(0)->type(), "f1"});

//	jive::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
//	jive::view(graph.root(), stdout);
}

static int
verify()
{
  TestRoot();
  TestGamma();
  TestGamma2();
  TestTheta();
  TestNestedTheta();
  TestEvolvingTheta();
  TestLambda();
  TestPhi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/TestDeadNodeElimination", verify)
