/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/CommonNodeElimination.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static jlm::util::StatisticsCollector statisticsCollector;

TEST(CommonNodeEliminationTests, test_simple)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, vt, "z");

  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), {}, { vt })->output(0);
  auto n2 = TestOperation::createNode(&graph.GetRootRegion(), {}, { vt })->output(0);

  auto u1 = TestOperation::createNode(&graph.GetRootRegion(), { z }, { vt })->output(0);

  auto b1 = TestOperation::createNode(&graph.GetRootRegion(), { x, y }, { vt })->output(0);
  auto b2 = TestOperation::createNode(&graph.GetRootRegion(), { x, y }, { vt })->output(0);
  auto b3 = TestOperation::createNode(&graph.GetRootRegion(), { n1, z }, { vt })->output(0);
  auto b4 = TestOperation::createNode(&graph.GetRootRegion(), { n2, z }, { vt })->output(0);

  jlm::rvsdg::GraphExport::Create(*n1, "n1");
  jlm::rvsdg::GraphExport::Create(*n2, "n2");
  jlm::rvsdg::GraphExport::Create(*u1, "u1");
  jlm::rvsdg::GraphExport::Create(*b1, "b1");
  jlm::rvsdg::GraphExport::Create(*b2, "b2");
  jlm::rvsdg::GraphExport::Create(*b3, "b3");
  jlm::rvsdg::GraphExport::Create(*b4, "b4");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  EXPECT_EQ(graph.GetRootRegion().result(0)->origin(), graph.GetRootRegion().result(1)->origin());
  EXPECT_EQ(graph.GetRootRegion().result(3)->origin(), graph.GetRootRegion().result(4)->origin());
  EXPECT_EQ(graph.GetRootRegion().result(5)->origin(), graph.GetRootRegion().result(6)->origin());
}

TEST(CommonNodeEliminationTests, test_gamma)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, vt, "z");

  auto u1 = TestOperation::createNode(&graph.GetRootRegion(), { x }, { vt })->output(0);
  auto u2 = TestOperation::createNode(&graph.GetRootRegion(), { x }, { vt })->output(0);

  auto gamma = jlm::rvsdg::GammaNode::create(c, 2);

  auto ev1 = gamma->AddEntryVar(u1);
  auto ev2 = gamma->AddEntryVar(u2);
  auto ev3 = gamma->AddEntryVar(y);
  auto ev4 = gamma->AddEntryVar(z);
  auto ev5 = gamma->AddEntryVar(z);

  auto n1 = TestOperation::createNode(gamma->subregion(0), {}, { vt })->output(0);
  auto n2 = TestOperation::createNode(gamma->subregion(0), {}, { vt })->output(0);
  auto n3 = TestOperation::createNode(gamma->subregion(0), {}, { vt })->output(0);

  gamma->AddExitVar({ ev1.branchArgument[0], ev1.branchArgument[1] });
  gamma->AddExitVar({ ev2.branchArgument[0], ev2.branchArgument[1] });
  gamma->AddExitVar({ ev3.branchArgument[0], ev3.branchArgument[1] });
  gamma->AddExitVar({ n1, ev3.branchArgument[1] });
  gamma->AddExitVar({ n2, ev3.branchArgument[1] });
  gamma->AddExitVar({ n3, ev3.branchArgument[1] });
  gamma->AddExitVar({ ev5.branchArgument[0], ev4.branchArgument[1] });

  jlm::rvsdg::GraphExport::Create(*gamma->output(0), "x1");
  jlm::rvsdg::GraphExport::Create(*gamma->output(1), "x2");
  jlm::rvsdg::GraphExport::Create(*gamma->output(2), "y");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  auto subregion0 = gamma->subregion(0);
  auto subregion1 = gamma->subregion(1);
  EXPECT_EQ(gamma->input(1)->origin(), gamma->input(2)->origin());
  EXPECT_EQ(subregion0->result(0)->origin(), subregion0->result(1)->origin());
  EXPECT_EQ(subregion0->result(3)->origin(), subregion0->result(4)->origin());
  EXPECT_EQ(subregion0->result(3)->origin(), subregion0->result(5)->origin());
  EXPECT_EQ(subregion1->result(0)->origin(), subregion1->result(1)->origin());
  EXPECT_EQ(graph.GetRootRegion().result(0)->origin(), graph.GetRootRegion().result(1)->origin());

  auto argument0 =
      dynamic_cast<const jlm::rvsdg::RegionArgument *>(subregion0->result(6)->origin());
  auto argument1 =
      dynamic_cast<const jlm::rvsdg::RegionArgument *>(subregion1->result(6)->origin());
  EXPECT_EQ(argument0->input(), argument1->input());
}

TEST(CommonNodeEliminationTests, test_theta)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto region = theta->subregion();

  auto lv1 = theta->AddLoopVar(c);
  auto lv2 = theta->AddLoopVar(x);
  auto lv3 = theta->AddLoopVar(x);
  auto lv4 = theta->AddLoopVar(x);

  auto u1 = TestOperation::createNode(region, { lv2.pre }, { vt })->output(0);
  auto u2 = TestOperation::createNode(region, { lv3.pre }, { vt })->output(0);
  auto b1 = TestOperation::createNode(region, { lv3.pre, lv4.pre }, { vt })->output(0);

  lv2.post->divert_to(u1);
  lv3.post->divert_to(u2);
  lv4.post->divert_to(b1);

  theta->set_predicate(lv1.pre);

  jlm::rvsdg::GraphExport::Create(*lv2.output, "lv2");
  jlm::rvsdg::GraphExport::Create(*lv3.output, "lv3");
  jlm::rvsdg::GraphExport::Create(*lv4.output, "lv4");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  auto un1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*u1);
  auto un2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*u2);
  auto bn1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*b1);
  EXPECT_EQ(un1->input(0)->origin(), un2->input(0)->origin());
  EXPECT_EQ(bn1->input(0)->origin(), un1->input(0)->origin());
  EXPECT_EQ(bn1->input(1)->origin(), region->argument(3));
  EXPECT_EQ(region->result(2)->origin(), region->result(3)->origin());
  EXPECT_EQ(graph.GetRootRegion().result(0)->origin(), graph.GetRootRegion().result(1)->origin());
}

TEST(CommonNodeEliminationTests, test_theta2)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto region = theta->subregion();

  auto lv1 = theta->AddLoopVar(c);
  auto lv2 = theta->AddLoopVar(x);
  auto lv3 = theta->AddLoopVar(x);

  auto u1 = TestOperation::createNode(region, { lv2.pre }, { vt })->output(0);
  auto u2 = TestOperation::createNode(region, { lv3.pre }, { vt })->output(0);
  auto b1 = TestOperation::createNode(region, { u2, u2 }, { vt })->output(0);

  lv2.post->divert_to(u1);
  lv3.post->divert_to(b1);

  theta->set_predicate(lv1.pre);

  jlm::rvsdg::GraphExport::Create(*lv2.output, "lv2");
  jlm::rvsdg::GraphExport::Create(*lv3.output, "lv3");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  EXPECT_EQ(lv2.post->origin(), u1);
  EXPECT_NE(lv2.pre->nusers(), 0u);
  EXPECT_NE(lv3.pre->nusers(), 0u);
}

TEST(CommonNodeEliminationTests, test_theta3)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");

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

  auto u1 = TestOperation::createNode(r1, { p2.output }, { vt });
  auto b1 = TestOperation::createNode(r1, { p3.output, p3.output }, { vt });
  auto u2 = TestOperation::createNode(r1, { p4.output }, { vt });

  lv2.post->divert_to(u1->output(0));
  lv3.post->divert_to(b1->output(0));
  lv4.post->divert_to(u1->output(0));

  theta1->set_predicate(lv1.pre);

  jlm::rvsdg::GraphExport::Create(*lv2.output, "lv2");
  jlm::rvsdg::GraphExport::Create(*lv3.output, "lv3");
  jlm::rvsdg::GraphExport::Create(*lv4.output, "lv4");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  EXPECT_EQ(r1->result(2)->origin(), r1->result(4)->origin());
  EXPECT_EQ(u1->input(0)->origin(), u2->input(0)->origin());
  EXPECT_EQ(r2->result(2)->origin(), r2->result(4)->origin());
  EXPECT_EQ(theta2->input(1)->origin(), theta2->input(3)->origin());
  EXPECT_NE(r1->result(3)->origin(), r1->result(4)->origin());
  EXPECT_NE(r2->result(3)->origin(), r2->result(4)->origin());
}

TEST(CommonNodeEliminationTests, test_theta4)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, vt, "y");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto region = theta->subregion();

  auto lv1 = theta->AddLoopVar(c);
  auto lv2 = theta->AddLoopVar(x);
  auto lv3 = theta->AddLoopVar(x);
  auto lv4 = theta->AddLoopVar(y);
  auto lv5 = theta->AddLoopVar(y);
  auto lv6 = theta->AddLoopVar(x);
  auto lv7 = theta->AddLoopVar(x);

  auto u1 = TestOperation::createNode(region, { lv2.pre }, { vt });
  auto b1 = TestOperation::createNode(region, { lv3.pre, lv3.pre }, { vt });

  lv2.post->divert_to(lv4.pre);
  lv3.post->divert_to(lv5.pre);
  lv4.post->divert_to(u1->output(0));
  lv5.post->divert_to(b1->output(0));

  theta->set_predicate(lv1.pre);

  auto & ex1 = jlm::rvsdg::GraphExport::Create(*theta->output(1), "lv2");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*theta->output(2), "lv3");
  jlm::rvsdg::GraphExport::Create(*theta->output(3), "lv4");
  jlm::rvsdg::GraphExport::Create(*theta->output(4), "lv5");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  EXPECT_NE(ex1.origin(), ex2.origin());
  EXPECT_NE(lv2.pre->nusers(), 0u);
  EXPECT_NE(lv3.pre->nusers(), 0u);
  EXPECT_EQ(lv6.post->origin(), lv7.post->origin());
}

TEST(CommonNodeEliminationTests, test_theta5)
{
  using namespace jlm::llvm;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, vt, "y");

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

  auto & ex1 = jlm::rvsdg::GraphExport::Create(*theta->output(1), "lv1");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*theta->output(2), "lv2");
  auto & ex3 = jlm::rvsdg::GraphExport::Create(*theta->output(3), "lv3");
  auto & ex4 = jlm::rvsdg::GraphExport::Create(*theta->output(4), "lv4");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  EXPECT_EQ(ex1.origin(), ex2.origin());
  EXPECT_EQ(ex3.origin(), ex4.origin());
  EXPECT_EQ(region->result(4)->origin(), region->result(5)->origin());
  EXPECT_EQ(region->result(2)->origin(), region->result(3)->origin());
}

TEST(CommonNodeEliminationTests, MultipleThetas)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & i0 = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "i0");

  // Loop 1
  auto thetaNode1 = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVariable1 = thetaNode1->AddLoopVar(&i0);
  auto node1 =
      TestOperation::createNode(thetaNode1->subregion(), { loopVariable1.pre }, { valueType });
  loopVariable1.post->divert_to(node1->output(0));

  // Loop 2
  auto thetaNode2 = ThetaNode::create(&rvsdg.GetRootRegion());
  auto predicate = &ControlConstantOperation::create(*thetaNode2->subregion(), 2, 1);
  thetaNode2->set_predicate(predicate);
  auto loopVariable2 = thetaNode2->AddLoopVar(&i0);
  auto node2 =
      TestOperation::createNode(thetaNode1->subregion(), { loopVariable2.pre }, { valueType });
  loopVariable2.post->divert_to(node2->output(0));

  // Loop 3
  auto thetaNode3 = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVariable3 = thetaNode3->AddLoopVar(loopVariable1.output);
  auto loopVariable4 = thetaNode3->AddLoopVar(loopVariable2.output);

  auto & x1 = jlm::rvsdg::GraphExport::Create(*loopVariable3.output, "x1");
  auto & x2 = jlm::rvsdg::GraphExport::Create(*loopVariable4.output, "x2");

  view(rvsdg, stdout);

  // Act
  CommonNodeElimination commonNodeElimination;
  commonNodeElimination.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // The origins from x1 and x2 are ultimately from two different loops with different iteration
  // counts. They are NOT congruent.
  EXPECT_NE(x1.origin(), x2.origin());
}

TEST(CommonNodeEliminationTests, MultipleThetasPassthrough)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & i0 = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "i0");

  // Loop 1
  auto thetaNode1 = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVariable1 = thetaNode1->AddLoopVar(&i0);

  // Loop 2
  auto thetaNode2 = ThetaNode::create(&rvsdg.GetRootRegion());
  auto predicate = &ControlConstantOperation::create(*thetaNode2->subregion(), 2, 1);
  thetaNode2->set_predicate(predicate);
  auto loopVariable2 = thetaNode2->AddLoopVar(&i0);

  // Loop 3
  auto thetaNode3 = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVariable3 = thetaNode3->AddLoopVar(loopVariable1.output);
  auto loopVariable4 = thetaNode3->AddLoopVar(loopVariable2.output);

  auto & x1 = jlm::rvsdg::GraphExport::Create(*loopVariable3.output, "x1");
  auto & x2 = jlm::rvsdg::GraphExport::Create(*loopVariable4.output, "x2");

  view(rvsdg, stdout);

  // Act
  CommonNodeElimination commonNodeElimination;
  commonNodeElimination.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // The origins from x1 and x2 are ultimately from two different loops with different iteration
  // counts, BUT the values in these loops are only passthrough values. Thus, we would expect them
  // to be congruent.
  EXPECT_EQ(x1.origin(), x2.origin());
}

TEST(CommonNodeEliminationTests, test_lambda)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ft = jlm::rvsdg::FunctionType::Create({ vt, vt }, { vt });

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", Linkage::externalLinkage));

  auto d1 = lambda->AddContextVar(*x).inner;
  auto d2 = lambda->AddContextVar(*x).inner;

  auto b1 = TestOperation::createNode(lambda->subregion(), { d1, d2 }, { vt })->output(0);

  auto output = lambda->finalize({ b1 });

  jlm::rvsdg::GraphExport::Create(*output, "f");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  auto bn1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*b1);
  EXPECT_EQ(bn1->input(0)->origin(), bn1->input(1)->origin());
}

TEST(CommonNodeEliminationTests, test_phi)
{
  using namespace jlm::llvm;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ft = jlm::rvsdg::FunctionType::Create({ vt, vt }, { vt });

  LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto & x = jlm::rvsdg::GraphImport::Create(graph, vt, "x");

  jlm::rvsdg::PhiBuilder pb;
  pb.begin(&graph.GetRootRegion());
  auto region = pb.subregion();

  auto d1 = pb.AddContextVar(x);
  auto d2 = pb.AddContextVar(x);

  auto r1 = pb.AddFixVar(ft);
  auto r2 = pb.AddFixVar(ft);

  auto lambda1 = jlm::rvsdg::LambdaNode::Create(
      *region,
      LlvmLambdaOperation::Create(ft, "f", Linkage::externalLinkage));
  auto cv1 = lambda1->AddContextVar(*d1.inner).inner;
  auto f1 = lambda1->finalize({ cv1 });

  auto lambda2 = jlm::rvsdg::LambdaNode::Create(
      *region,
      LlvmLambdaOperation::Create(ft, "f", Linkage::externalLinkage));
  auto cv2 = lambda2->AddContextVar(*d2.inner).inner;
  auto f2 = lambda2->finalize({ cv2 });

  r1.result->divert_to(f1);
  r2.result->divert_to(f2);

  auto phi = pb.end();

  jlm::rvsdg::GraphExport::Create(*phi->output(0), "f1");
  jlm::rvsdg::GraphExport::Create(*phi->output(1), "f2");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::CommonNodeElimination cne;
  cne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  EXPECT_EQ(
      jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*f1).input(0)->origin(),
      jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*f2).input(0)->origin());
}

TEST(CommonNodeEliminationTests, EmptyTheta)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();
  auto controlType = ControlType::Create(2);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto node1 = TestOperation::createNode(thetaNode->subregion(), {}, { valueType });
  auto node2 =
      TestOperation::createNode(thetaNode->subregion(), { node1->output(0) }, { valueType });
  auto node3 =
      TestOperation::createNode(thetaNode->subregion(), { node1->output(0) }, { valueType });
  auto node4 = TestOperation::createNode(
      thetaNode->subregion(),
      { node2->output(0), node3->output(0) },
      { controlType });

  thetaNode->set_predicate(node4->output(0));

  view(rvsdg, stdout);

  // Act
  CommonNodeElimination commonNodeElimination;
  commonNodeElimination.Run(rvsdgModule, statisticsCollector);

  thetaNode->subregion()->prune(false);

  view(rvsdg, stdout);

  // Assert
  // We expect that node2 and node3 are unified in the theta subregion
  EXPECT_EQ(thetaNode->subregion()->numNodes(), 3u);
}
