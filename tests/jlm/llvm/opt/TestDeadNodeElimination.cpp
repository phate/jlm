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

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Phi.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunDeadNodeElimination(jlm::llvm::RvsdgModule & rvsdgModule)
{
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::DeadNodeElimination deadNodeElimination;
  deadNodeElimination.run(rvsdgModule, statisticsCollector);
}

static void
TestRoot()
{
  using namespace jlm::llvm;

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  jlm::tests::GraphImport::Create(graph, jlm::tests::valuetype::Create(), "x");
  auto y = &jlm::tests::GraphImport::Create(graph, jlm::tests::valuetype::Create(), "y");
  GraphExport::Create(*y, "z");

  //	jlm::rvsdg::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
  //	jlm::rvsdg::view(graph.root(), stdout);

  assert(graph.root()->narguments() == 1);
}

static void
TestGamma()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");

  auto gamma = jlm::rvsdg::gamma_node::create(c, 2);
  auto ev1 = gamma->add_entryvar(x);
  auto ev2 = gamma->add_entryvar(y);
  auto ev3 = gamma->add_entryvar(x);

  auto t = jlm::tests::create_testop(gamma->subregion(1), { ev2->argument(1) }, { vt })[0];

  gamma->add_exitvar({ ev1->argument(0), ev1->argument(1) });
  gamma->add_exitvar({ ev2->argument(0), t });
  gamma->add_exitvar({ ev3->argument(0), ev1->argument(1) });

  GraphExport::Create(*gamma->output(0), "z");
  GraphExport::Create(*gamma->output(2), "w");

  //	jlm::rvsdg::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
  //	jlm::rvsdg::view(graph.root(), stdout);

  assert(gamma->noutputs() == 2);
  assert(gamma->subregion(1)->nodes.empty());
  assert(gamma->subregion(1)->narguments() == 2);
  assert(gamma->ninputs() == 3);
  assert(graph.root()->narguments() == 2);
}

static void
TestGamma2()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto gamma = jlm::rvsdg::gamma_node::create(c, 2);
  gamma->add_entryvar(x);

  auto n1 = jlm::tests::create_testop(gamma->subregion(0), {}, { vt })[0];
  auto n2 = jlm::tests::create_testop(gamma->subregion(1), {}, { vt })[0];

  gamma->add_exitvar({ n1, n2 });

  GraphExport::Create(*gamma->output(0), "x");

  //	jlm::rvsdg::view(graph, stdout);
  RunDeadNodeElimination(rm);
  //	jlm::rvsdg::view(graph, stdout);

  assert(graph.root()->narguments() == 1);
}

static void
TestTheta()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::tests::GraphImport::Create(graph, vt, "z");

  auto theta = jlm::rvsdg::theta_node::create(graph.root());

  auto lv1 = theta->add_loopvar(x);
  auto lv2 = theta->add_loopvar(y);
  auto lv3 = theta->add_loopvar(z);
  auto lv4 = theta->add_loopvar(y);

  lv1->result()->divert_to(lv2->argument());
  lv2->result()->divert_to(lv1->argument());

  auto t = jlm::tests::create_testop(theta->subregion(), { lv3->argument() }, { vt })[0];
  lv3->result()->divert_to(t);
  lv4->result()->divert_to(lv2->argument());

  auto c = jlm::tests::create_testop(theta->subregion(), {}, { ct })[0];
  theta->set_predicate(c);

  GraphExport::Create(*theta->output(0), "a");
  GraphExport::Create(*theta->output(3), "b");

  //	jlm::rvsdg::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
  //	jlm::rvsdg::view(graph.root(), stdout);

  assert(theta->noutputs() == 3);
  assert(theta->subregion()->nodes.size() == 1);
  assert(graph.root()->narguments() == 2);
}

static void
TestNestedTheta()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");

  auto otheta = jlm::rvsdg::theta_node::create(graph.root());

  auto lvo1 = otheta->add_loopvar(c);
  auto lvo2 = otheta->add_loopvar(x);
  auto lvo3 = otheta->add_loopvar(y);

  auto itheta = jlm::rvsdg::theta_node::create(otheta->subregion());

  auto lvi1 = itheta->add_loopvar(lvo1->argument());
  auto lvi2 = itheta->add_loopvar(lvo2->argument());
  auto lvi3 = itheta->add_loopvar(lvo3->argument());

  lvi2->result()->divert_to(lvi3->argument());

  itheta->set_predicate(lvi1->argument());

  lvo2->result()->divert_to(itheta->output(1));
  lvo3->result()->divert_to(itheta->output(1));

  otheta->set_predicate(lvo1->argument());

  GraphExport::Create(*otheta->output(2), "y");

  //	jlm::rvsdg::view(graph, stdout);
  RunDeadNodeElimination(rm);
  //	jlm::rvsdg::view(graph, stdout);

  assert(otheta->noutputs() == 3);
}

static void
TestEvolvingTheta()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x1 = &jlm::tests::GraphImport::Create(graph, vt, "x1");
  auto x2 = &jlm::tests::GraphImport::Create(graph, vt, "x2");
  auto x3 = &jlm::tests::GraphImport::Create(graph, vt, "x3");
  auto x4 = &jlm::tests::GraphImport::Create(graph, vt, "x4");

  auto theta = jlm::rvsdg::theta_node::create(graph.root());

  auto lv0 = theta->add_loopvar(c);
  auto lv1 = theta->add_loopvar(x1);
  auto lv2 = theta->add_loopvar(x2);
  auto lv3 = theta->add_loopvar(x3);
  auto lv4 = theta->add_loopvar(x4);

  lv1->result()->divert_to(lv2->argument());
  lv2->result()->divert_to(lv3->argument());
  lv3->result()->divert_to(lv4->argument());

  theta->set_predicate(lv0->argument());

  GraphExport::Create(*lv1, "x1");

  //	jlm::rvsdg::view(graph, stdout);
  RunDeadNodeElimination(rm);
  //	jlm::rvsdg::view(graph, stdout);

  assert(theta->noutputs() == 5);
}

static void
TestLambda()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");

  auto lambda = lambda::node::create(
      graph.root(),
      FunctionType::Create({ vt }, { vt, vt }),
      "f",
      linkage::external_linkage);

  auto cv1 = lambda->add_ctxvar(x);
  auto cv2 = lambda->add_ctxvar(y);
  jlm::tests::create_testop(lambda->subregion(), { lambda->fctargument(0), cv1 }, { vt });

  auto output = lambda->finalize({ lambda->fctargument(0), cv2 });

  GraphExport::Create(*output, "f");

  //	jlm::rvsdg::view(graph.root(), stdout);
  RunDeadNodeElimination(rm);
  //	jlm::rvsdg::view(graph.root(), stdout);

  assert(lambda->subregion()->nodes.empty());
  assert(graph.root()->narguments() == 1);
}

static void
TestPhi()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::tests::GraphImport::Create(rvsdg, valueType, "z");

  auto setupF1 = [&](jlm::rvsdg::region & region, phi::rvoutput & rv2, jlm::rvsdg::argument & dx)
  {
    auto lambda1 = lambda::node::create(&region, functionType, "f1", linkage::external_linkage);
    auto f2Argument = lambda1->add_ctxvar(rv2.argument());
    auto xArgument = lambda1->add_ctxvar(&dx);

    auto result = jlm::tests::SimpleNode::Create(
                      *lambda1->subregion(),
                      { lambda1->fctargument(0), f2Argument, xArgument },
                      { valueType })
                      .output(0);

    return lambda1->finalize({ result });
  };

  auto setupF2 = [&](jlm::rvsdg::region & region, phi::rvoutput & rv1, jlm::rvsdg::argument & dy)
  {
    auto lambda2 = lambda::node::create(&region, functionType, "f2", linkage::external_linkage);
    auto f1Argument = lambda2->add_ctxvar(rv1.argument());
    lambda2->add_ctxvar(&dy);

    auto result = jlm::tests::SimpleNode::Create(
                      *lambda2->subregion(),
                      { lambda2->fctargument(0), f1Argument },
                      { valueType })
                      .output(0);

    return lambda2->finalize({ result });
  };

  auto setupF3 = [&](jlm::rvsdg::region & region, jlm::rvsdg::argument & dz)
  {
    auto lambda3 = lambda::node::create(&region, functionType, "f3", linkage::external_linkage);
    auto zArgument = lambda3->add_ctxvar(&dz);

    auto result = jlm::tests::SimpleNode::Create(
                      *lambda3->subregion(),
                      { lambda3->fctargument(0), zArgument },
                      { valueType })
                      .output(0);

    return lambda3->finalize({ result });
  };

  auto setupF4 = [&](jlm::rvsdg::region & region)
  {
    auto lambda = lambda::node::create(&region, functionType, "f4", linkage::external_linkage);
    return lambda->finalize({ lambda->fctargument(0) });
  };

  phi::builder phiBuilder;
  phiBuilder.begin(rvsdg.root());
  auto & phiSubregion = *phiBuilder.subregion();

  auto rv1 = phiBuilder.add_recvar(PointerType::Create());
  auto rv2 = phiBuilder.add_recvar(PointerType::Create());
  auto rv3 = phiBuilder.add_recvar(PointerType::Create());
  auto rv4 = phiBuilder.add_recvar(PointerType::Create());
  auto dx = phiBuilder.add_ctxvar(x);
  auto dy = phiBuilder.add_ctxvar(y);
  auto dz = phiBuilder.add_ctxvar(z);

  auto f1 = setupF1(phiSubregion, *rv2, *dx);
  auto f2 = setupF2(phiSubregion, *rv1, *dy);
  auto f3 = setupF3(phiSubregion, *dz);
  auto f4 = setupF4(phiSubregion);

  rv1->set_rvorigin(f1);
  rv2->set_rvorigin(f2);
  rv3->set_rvorigin(f3);
  rv4->set_rvorigin(f4);
  auto phiNode = phiBuilder.end();

  GraphExport::Create(*phiNode->output(0), "f1");
  GraphExport::Create(*phiNode->output(3), "f4");

  // Act
  RunDeadNodeElimination(rvsdgModule);

  // Assert
  assert(phiNode->noutputs() == 3); // f1, f2, and f4 are alive
  assert(phiNode->output(0) == rv1);
  assert(phiNode->output(1) == rv2);
  assert(phiNode->output(2) == rv4);
  assert(phiSubregion.nresults() == 3); // f1, f2, and f4 are alive
  assert(phiSubregion.result(0) == rv1->result());
  assert(phiSubregion.result(1) == rv2->result());
  assert(phiSubregion.result(2) == rv4->result());
  assert(phiSubregion.narguments() == 4); // f1, f2, f4, and dx are alive
  assert(phiSubregion.argument(0) == rv1->argument());
  assert(phiSubregion.argument(1) == rv2->argument());
  assert(phiSubregion.argument(2) == rv4->argument());
  assert(phiSubregion.argument(3) == dx);
  assert(phiNode->ninputs() == 1); // dx is alive
  assert(phiNode->input(0) == dx->input());
}

static void
TestDelta()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::tests::GraphImport::Create(rvsdg, valueType, "z");

  auto deltaNode =
      delta::node::Create(rvsdg.root(), valueType, "delta", linkage::external_linkage, "", false);

  auto xArgument = deltaNode->add_ctxvar(x);
  deltaNode->add_ctxvar(y);
  auto zArgument = deltaNode->add_ctxvar(z);

  auto result =
      jlm::tests::SimpleNode::Create(*deltaNode->subregion(), { xArgument }, { valueType })
          .output(0);

  jlm::tests::SimpleNode::Create(*deltaNode->subregion(), { zArgument }, {});

  auto deltaOutput = deltaNode->finalize(result);
  GraphExport::Create(*deltaOutput, "");

  // Act
  RunDeadNodeElimination(rvsdgModule);

  // Assert
  assert(deltaNode->subregion()->nnodes() == 1);
  assert(deltaNode->ninputs() == 1);
}

static int
TestDeadNodeElimination()
{
  TestRoot();
  TestGamma();
  TestGamma2();
  TestTheta();
  TestNestedTheta();
  TestEvolvingTheta();
  TestLambda();
  TestPhi();
  TestDelta();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/TestDeadNodeElimination", TestDeadNodeElimination)
