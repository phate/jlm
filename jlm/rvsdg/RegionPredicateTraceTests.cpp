/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include "jlm/rvsdg/bitstring/arithmetic.hpp"
#include "jlm/rvsdg/bitstring/constant.hpp"
#include "jlm/rvsdg/bitstring/type.hpp"
#include "jlm/rvsdg/simple-node.hpp"
#include "jlm/rvsdg/theta.hpp"
#include <gtest/gtest.h>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/RegionPredicateTrace.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(RegionPredicateTraceTests, TestTracing)
{
  using namespace jlm;

  auto valueType = rvsdg::TestType::createValueType();
  auto ctl2 = rvsdg::ControlType::Create(2);

  rvsdg::Graph rvsdg;
  auto & pred1 = rvsdg::GraphImport::Create(rvsdg, ctl2, "pred1");
  auto & pred2 = rvsdg::GraphImport::Create(rvsdg, ctl2, "pred2");

  // First gamma, computes a predicate
  auto gamma1 = rvsdg::GammaNode::create(&pred1, 2);
  auto g1_left = gamma1->subregion(0);
  auto g1_right = gamma1->subregion(1);
  auto & g1_p0 = rvsdg::ControlConstantOperation::createTrue(*g1_left);
  auto & g1_p1 = rvsdg::ControlConstantOperation::createFalse(*g1_right);
  auto pred3 = gamma1->AddExitVar({ &g1_p0, &g1_p1 }).output;

  // Second gamma, depends on that predicate.
  auto gamma2 = rvsdg::GammaNode::create(pred3, 2);
  auto g2_left = gamma2->subregion(0);
  auto g2_right = gamma2->subregion(1);
  auto r0 = rvsdg::TestOperation::createNode(g2_left, {}, { valueType });
  auto r1 = rvsdg::TestOperation::createNode(g2_right, {}, { valueType });
  auto r = gamma2->AddExitVar({ r0->output(0), r1->output(0) }).output;
  rvsdg::GraphExport::Create(*r, "result1");

  // Third gamma, depends on unrelated predicate.
  auto gamma3 = rvsdg::GammaNode::create(&pred2, 2);
  auto g3_left = gamma3->subregion(0);
  auto g3_right = gamma3->subregion(1);
  auto s0 = rvsdg::TestOperation::createNode(g3_left, {}, { valueType });
  auto s1 = rvsdg::TestOperation::createNode(g3_right, {}, { valueType });
  auto s = gamma3->AddExitVar({ s0->output(0), s1->output(0) }).output;
  rvsdg::GraphExport::Create(*s, "result2");

  rvsdg::RegionPredicateTrace trace;

  // Since gamma1 dominates gamma2, not all cross-paths are possible.
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_left, *g2_right));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_right, *g2_left));
  EXPECT_FALSE(trace.CheckPredicatesSatisfiable(*g1_left, *g2_left));
  EXPECT_FALSE(trace.CheckPredicatesSatisfiable(*g1_right, *g2_right));

  // Since gamma1 and gamma3 are unrelated,  all cross-paths are possible.
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_left, *g3_right));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_right, *g3_left));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_left, *g3_left));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_right, *g3_right));

  // Now change the graph, and check again.
  gamma2->predicate()->divert_to(&pred2);

  // Now, everything is uncorrelated.
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_left, *g2_right));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_right, *g2_left));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_left, *g2_left));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_right, *g2_right));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_left, *g3_right));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_right, *g3_left));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_left, *g3_left));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*g1_right, *g3_right));
}

TEST(RegionPredicateTraceTests, TraceOutOfTheta)
{
  /**
   * Creates an RVSDG graph that looks like:
   *
   * +-theta0------------------x-------------------------+
   * |                                                   |
   * | +-theta1---x-----+ +-theta3-------------x-------+ |
   * | |                | |                    v       | |
   * | | CTRL(0) INT(3) | |         +-theta4---x-----+ | |
   * | |   v      v     | |         |                | | |
   * | +----------x-----+ |         | CTRL(0) INT(7) | | |
   * |            v       |         |   v      v     | | |
   * | +-theta2---x-----+ | CTRL(0) +----------x-----+ | |
   * | | CTRL(0)  v     | |   v                v       | |
   * | +----------x-----+ +--------------------x-------+ |
   * |            |                            |         |
   * |            \----------\   /-------------/         |
   * |                        v v                        |
   * | CTRL(0)                ADD                        |
   * |   v                     v                         |
   * +-------------------------x-------------------------+
   *                           v
   *                       export("x")
   *
   * and checks that all regions are considered reachable from all regions above it,
   * both parent, child and sibling regions.
   */

  using namespace jlm;

  // Arrange
  auto bit32 = rvsdg::BitType::Create(32);

  rvsdg::Graph rvsdg;
  auto theta0 = rvsdg::ThetaNode::create(&rvsdg.GetRootRegion());
  auto undef0 =
      rvsdg::CreateOpNode<rvsdg::TestNullaryOperation>(rvsdg.GetRootRegion(), bit32).output(0);
  auto loopVar0 = theta0->AddLoopVar(undef0);

  // theta1
  auto theta1 = rvsdg::ThetaNode::create(theta0->subregion());
  auto undef1 =
      rvsdg::CreateOpNode<rvsdg::TestNullaryOperation>(*theta0->subregion(), bit32).output(0);
  auto loopVar1 = theta1->AddLoopVar(undef1);
  auto & int3Output = rvsdg::BitConstantOperation::create(*theta1->subregion(), { 32, 3 });
  loopVar1.post->divert_to(&int3Output);

  // theta2
  auto theta2 = rvsdg::ThetaNode::create(theta0->subregion());
  auto loopVar2 = theta2->AddLoopVar(loopVar1.output);

  // theta3
  auto theta3 = rvsdg::ThetaNode::create(theta0->subregion());
  auto undef3 =
      rvsdg::CreateOpNode<rvsdg::TestNullaryOperation>(*theta0->subregion(), bit32).output(0);
  auto loopVar3 = theta3->AddLoopVar(undef3);

  // theta4
  auto theta4 = rvsdg::ThetaNode::create(theta3->subregion());
  auto loopVar4 = theta4->AddLoopVar(loopVar3.pre);
  auto & int7Output = rvsdg::BitConstantOperation::create(*theta4->subregion(), { 32, 7 });
  loopVar4.post->divert_to(&int7Output);
  loopVar3.post->divert_to(loopVar4.output);

  // ADD inside theta0's subregion, combining outputs from theta2 and theta3/theta4
  auto & addOutput = *rvsdg::bitadd_op::create(32, loopVar2.output, loopVar3.output);
  loopVar0.post->divert_to(&addOutput);

  rvsdg::GraphExport::Create(*loopVar0.output, "x");

  // Assert
  rvsdg::RegionPredicateTrace trace;

  // Every region can be reached from the root region
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(rvsdg.GetRootRegion(), *theta0->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(rvsdg.GetRootRegion(), *theta1->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(rvsdg.GetRootRegion(), *theta2->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(rvsdg.GetRootRegion(), *theta3->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(rvsdg.GetRootRegion(), *theta4->subregion()));

  // Every region can reach the root region
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta0->subregion(), rvsdg.GetRootRegion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta1->subregion(), rvsdg.GetRootRegion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta1->subregion(), rvsdg.GetRootRegion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta1->subregion(), rvsdg.GetRootRegion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta1->subregion(), rvsdg.GetRootRegion()));

  // theta0 can reach every region inside it
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta0->subregion(), *theta1->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta0->subregion(), *theta2->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta0->subregion(), *theta3->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta0->subregion(), *theta4->subregion()));

  // theta0 can also be reached by every region inside it
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta1->subregion(), *theta0->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta2->subregion(), *theta0->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta3->subregion(), *theta0->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta4->subregion(), *theta0->subregion()));

  // theta2 can be reached from theta1
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta1->subregion(), *theta2->subregion()));

  // theta3 and theta4 can reach each other
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta3->subregion(), *theta4->subregion()));
  EXPECT_TRUE(trace.CheckPredicatesSatisfiable(*theta4->subregion(), *theta3->subregion()));
}
