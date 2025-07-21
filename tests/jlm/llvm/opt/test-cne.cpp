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
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/cne.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/util/Statistics.hpp>

static jlm::util::StatisticsCollector statisticsCollector;

static inline void
test_simple()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::tests::GraphImport::Create(graph, vt, "z");

  auto n1 = jlm::tests::create_testop(&graph.GetRootRegion(), {}, { vt })[0];
  auto n2 = jlm::tests::create_testop(&graph.GetRootRegion(), {}, { vt })[0];

  auto u1 = jlm::tests::create_testop(&graph.GetRootRegion(), { z }, { vt })[0];

  auto b1 = jlm::tests::create_testop(&graph.GetRootRegion(), { x, y }, { vt })[0];
  auto b2 = jlm::tests::create_testop(&graph.GetRootRegion(), { x, y }, { vt })[0];
  auto b3 = jlm::tests::create_testop(&graph.GetRootRegion(), { n1, z }, { vt })[0];
  auto b4 = jlm::tests::create_testop(&graph.GetRootRegion(), { n2, z }, { vt })[0];

  GraphExport::Create(*n1, "n1");
  GraphExport::Create(*n2, "n2");
  GraphExport::Create(*u1, "u1");
  GraphExport::Create(*b1, "b1");
  GraphExport::Create(*b2, "b2");
  GraphExport::Create(*b3, "b3");
  GraphExport::Create(*b4, "b4");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  assert(graph.GetRootRegion().result(0)->origin() == graph.GetRootRegion().result(1)->origin());
  assert(graph.GetRootRegion().result(3)->origin() == graph.GetRootRegion().result(4)->origin());
  assert(graph.GetRootRegion().result(5)->origin() == graph.GetRootRegion().result(6)->origin());
}

static inline void
test_gamma()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::tests::GraphImport::Create(graph, vt, "z");

  auto u1 = jlm::tests::create_testop(&graph.GetRootRegion(), { x }, { vt })[0];
  auto u2 = jlm::tests::create_testop(&graph.GetRootRegion(), { x }, { vt })[0];

  auto gamma = jlm::rvsdg::GammaNode::create(c, 2);

  auto ev1 = gamma->AddEntryVar(u1);
  auto ev2 = gamma->AddEntryVar(u2);
  auto ev3 = gamma->AddEntryVar(y);
  auto ev4 = gamma->AddEntryVar(z);
  auto ev5 = gamma->AddEntryVar(z);

  auto n1 = jlm::tests::create_testop(gamma->subregion(0), {}, { vt })[0];
  auto n2 = jlm::tests::create_testop(gamma->subregion(0), {}, { vt })[0];
  auto n3 = jlm::tests::create_testop(gamma->subregion(0), {}, { vt })[0];

  gamma->AddExitVar({ ev1.branchArgument[0], ev1.branchArgument[1] });
  gamma->AddExitVar({ ev2.branchArgument[0], ev2.branchArgument[1] });
  gamma->AddExitVar({ ev3.branchArgument[0], ev3.branchArgument[1] });
  gamma->AddExitVar({ n1, ev3.branchArgument[1] });
  gamma->AddExitVar({ n2, ev3.branchArgument[1] });
  gamma->AddExitVar({ n3, ev3.branchArgument[1] });
  gamma->AddExitVar({ ev5.branchArgument[0], ev4.branchArgument[1] });

  GraphExport::Create(*gamma->output(0), "x1");
  GraphExport::Create(*gamma->output(1), "x2");
  GraphExport::Create(*gamma->output(2), "y");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  auto subregion0 = gamma->subregion(0);
  auto subregion1 = gamma->subregion(1);
  assert(gamma->input(1)->origin() == gamma->input(2)->origin());
  assert(subregion0->result(0)->origin() == subregion0->result(1)->origin());
  assert(subregion0->result(3)->origin() == subregion0->result(4)->origin());
  assert(subregion0->result(3)->origin() == subregion0->result(5)->origin());
  assert(subregion1->result(0)->origin() == subregion1->result(1)->origin());
  assert(graph.GetRootRegion().result(0)->origin() == graph.GetRootRegion().result(1)->origin());

  auto argument0 =
      dynamic_cast<const jlm::rvsdg::RegionArgument *>(subregion0->result(6)->origin());
  auto argument1 =
      dynamic_cast<const jlm::rvsdg::RegionArgument *>(subregion1->result(6)->origin());
  assert(argument0->input() == argument1->input());
}

static inline void
test_theta()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto region = theta->subregion();

  auto lv1 = theta->AddLoopVar(c);
  auto lv2 = theta->AddLoopVar(x);
  auto lv3 = theta->AddLoopVar(x);
  auto lv4 = theta->AddLoopVar(x);

  auto u1 = jlm::tests::create_testop(region, { lv2.pre }, { vt })[0];
  auto u2 = jlm::tests::create_testop(region, { lv3.pre }, { vt })[0];
  auto b1 = jlm::tests::create_testop(region, { lv3.pre, lv4.pre }, { vt })[0];

  lv2.post->divert_to(u1);
  lv3.post->divert_to(u2);
  lv4.post->divert_to(b1);

  theta->set_predicate(lv1.pre);

  GraphExport::Create(*lv2.output, "lv2");
  GraphExport::Create(*lv3.output, "lv3");
  GraphExport::Create(*lv4.output, "lv4");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  auto un1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*u1);
  auto un2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*u2);
  auto bn1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*b1);
  assert(un1->input(0)->origin() == un2->input(0)->origin());
  assert(bn1->input(0)->origin() == un1->input(0)->origin());
  assert(bn1->input(1)->origin() == region->argument(3));
  assert(region->result(2)->origin() == region->result(3)->origin());
  assert(graph.GetRootRegion().result(0)->origin() == graph.GetRootRegion().result(1)->origin());
}

static inline void
test_theta2()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto region = theta->subregion();

  auto lv1 = theta->AddLoopVar(c);
  auto lv2 = theta->AddLoopVar(x);
  auto lv3 = theta->AddLoopVar(x);

  auto u1 = jlm::tests::create_testop(region, { lv2.pre }, { vt })[0];
  auto u2 = jlm::tests::create_testop(region, { lv3.pre }, { vt })[0];
  auto b1 = jlm::tests::create_testop(region, { u2, u2 }, { vt })[0];

  lv2.post->divert_to(u1);
  lv3.post->divert_to(b1);

  theta->set_predicate(lv1.pre);

  GraphExport::Create(*lv2.output, "lv2");
  GraphExport::Create(*lv3.output, "lv3");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  assert(lv2.post->origin() == u1);
  assert(lv2.pre->nusers() != 0 && lv3.pre->nusers() != 0);
}

static inline void
test_theta3()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto theta1 = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto r1 = theta1->subregion();

  auto lv1 = theta1->AddLoopVar(c);
  auto lv2 = theta1->AddLoopVar(x);
  auto lv3 = theta1->AddLoopVar(x);
  auto lv4 = theta1->AddLoopVar(x);

  auto theta2 = jlm::rvsdg::ThetaNode::create(r1);
  auto r2 = theta2->subregion();
  auto p = theta2->AddLoopVar(lv1.pre);
  auto p2 = theta2->AddLoopVar(lv2.pre);
  auto p3 = theta2->AddLoopVar(lv3.pre);
  auto p4 = theta2->AddLoopVar(lv4.pre);
  theta2->set_predicate(p.pre);

  auto u1 = jlm::tests::TestOperation::create(r1, { p2.output }, { vt });
  auto b1 = jlm::tests::TestOperation::create(r1, { p3.output, p3.output }, { vt });
  auto u2 = jlm::tests::TestOperation::create(r1, { p4.output }, { vt });

  lv2.post->divert_to(u1->output(0));
  lv3.post->divert_to(b1->output(0));
  lv4.post->divert_to(u1->output(0));

  theta1->set_predicate(lv1.pre);

  GraphExport::Create(*lv2.output, "lv2");
  GraphExport::Create(*lv3.output, "lv3");
  GraphExport::Create(*lv4.output, "lv4");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  assert(r1->result(2)->origin() == r1->result(4)->origin());
  assert(u1->input(0)->origin() == u2->input(0)->origin());
  assert(r2->result(2)->origin() == r2->result(4)->origin());
  assert(theta2->input(1)->origin() == theta2->input(3)->origin());
  assert(r1->result(3)->origin() != r1->result(4)->origin());
  assert(r2->result(3)->origin() != r2->result(4)->origin());
}

static inline void
test_theta4()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto region = theta->subregion();

  auto lv1 = theta->AddLoopVar(c);
  auto lv2 = theta->AddLoopVar(x);
  auto lv3 = theta->AddLoopVar(x);
  auto lv4 = theta->AddLoopVar(y);
  auto lv5 = theta->AddLoopVar(y);
  auto lv6 = theta->AddLoopVar(x);
  auto lv7 = theta->AddLoopVar(x);

  auto u1 = jlm::tests::TestOperation::create(region, { lv2.pre }, { vt });
  auto b1 = jlm::tests::TestOperation::create(region, { lv3.pre, lv3.pre }, { vt });

  lv2.post->divert_to(lv4.pre);
  lv3.post->divert_to(lv5.pre);
  lv4.post->divert_to(u1->output(0));
  lv5.post->divert_to(b1->output(0));

  theta->set_predicate(lv1.pre);

  auto & ex1 = GraphExport::Create(*theta->output(1), "lv2");
  auto & ex2 = GraphExport::Create(*theta->output(2), "lv3");
  GraphExport::Create(*theta->output(3), "lv4");
  GraphExport::Create(*theta->output(4), "lv5");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  assert(ex1.origin() != ex2.origin());
  assert(lv2.pre->nusers() != 0 && lv3.pre->nusers() != 0);
  assert(lv6.post->origin() == lv7.post->origin());
}

static inline void
test_theta5()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto region = theta->subregion();

  auto lv0 = theta->AddLoopVar(c);
  auto lv1 = theta->AddLoopVar(x);
  auto lv2 = theta->AddLoopVar(x);
  auto lv3 = theta->AddLoopVar(y);
  auto lv4 = theta->AddLoopVar(y);

  lv1.post->divert_to(lv3.pre);
  lv2.post->divert_to(lv4.pre);

  theta->set_predicate(lv0.pre);

  auto & ex1 = GraphExport::Create(*theta->output(1), "lv1");
  auto & ex2 = GraphExport::Create(*theta->output(2), "lv2");
  auto & ex3 = GraphExport::Create(*theta->output(3), "lv3");
  auto & ex4 = GraphExport::Create(*theta->output(4), "lv4");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  assert(ex1.origin() == ex2.origin());
  assert(ex3.origin() == ex4.origin());
  assert(region->result(4)->origin() == region->result(5)->origin());
  assert(region->result(2)->origin() == region->result(3)->origin());
}

static inline void
test_lambda()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();
  auto ft = jlm::rvsdg::FunctionType::Create({ vt, vt }, { vt });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", linkage::external_linkage));

  auto d1 = lambda->AddContextVar(*x).inner;
  auto d2 = lambda->AddContextVar(*x).inner;

  auto b1 = jlm::tests::create_testop(lambda->subregion(), { d1, d2 }, { vt })[0];

  auto output = lambda->finalize({ b1 });

  GraphExport::Create(*output, "f");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  auto bn1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*b1);
  assert(bn1->input(0)->origin() == bn1->input(1)->origin());
}

static inline void
test_phi()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::ValueType::Create();
  auto ft = jlm::rvsdg::FunctionType::Create({ vt, vt }, { vt });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto & x = jlm::tests::GraphImport::Create(graph, vt, "x");

  jlm::rvsdg::PhiBuilder pb;
  pb.begin(&graph.GetRootRegion());
  auto region = pb.subregion();

  auto d1 = pb.AddContextVar(x);
  auto d2 = pb.AddContextVar(x);

  auto r1 = pb.AddFixVar(ft);
  auto r2 = pb.AddFixVar(ft);

  auto lambda1 = jlm::rvsdg::LambdaNode::Create(
      *region,
      LlvmLambdaOperation::Create(ft, "f", linkage::external_linkage));
  auto cv1 = lambda1->AddContextVar(*d1.inner).inner;
  auto f1 = lambda1->finalize({ cv1 });

  auto lambda2 = jlm::rvsdg::LambdaNode::Create(
      *region,
      LlvmLambdaOperation::Create(ft, "f", linkage::external_linkage));
  auto cv2 = lambda2->AddContextVar(*d2.inner).inner;
  auto f2 = lambda2->finalize({ cv2 });

  r1.result->divert_to(f1);
  r2.result->divert_to(f2);

  auto phi = pb.end();

  GraphExport::Create(*phi->output(0), "f1");
  GraphExport::Create(*phi->output(1), "f2");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  assert(
      jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*f1).input(0)->origin()
      == jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*f2).input(0)->origin());
}

static void
verify()
{
  test_simple();
  test_gamma();
  test_theta();
  test_theta2();
  test_theta3();
  test_theta4();
  test_theta5();
  test_lambda();
  test_phi();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-cne", verify)
