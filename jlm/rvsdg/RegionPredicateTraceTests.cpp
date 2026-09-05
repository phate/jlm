/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/RegionPredicateTrace.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>

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

  rvsdg::AlternativeRegionPredicateTracer trace;

  // Since gamma1 dominates gamma2, not all cross-paths are possible.
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_left, *g2_right));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_right, *g2_left));
  EXPECT_FALSE(trace.canRegionReachRegion(*g1_left, *g2_left));
  EXPECT_FALSE(trace.canRegionReachRegion(*g1_right, *g2_right));

  // Since gamma1 and gamma3 are unrelated,  all cross-paths are possible.
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_left, *g3_right));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_right, *g3_left));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_left, *g3_left));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_right, *g3_right));

  // Now change the graph, and check again.
  gamma2->predicate()->divert_to(&pred2);
  trace.clearCaches();

  // Now, everything is uncorrelated.
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_left, *g2_right));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_right, *g2_left));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_left, *g2_left));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_right, *g2_right));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_left, *g3_right));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_right, *g3_left));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_left, *g3_left));
  EXPECT_TRUE(trace.canRegionReachRegion(*g1_right, *g3_right));
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
  rvsdg::AlternativeRegionPredicateTracer trace;

  // Every region can be reached from the root region
  EXPECT_TRUE(trace.canRegionReachRegion(rvsdg.GetRootRegion(), *theta0->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(rvsdg.GetRootRegion(), *theta1->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(rvsdg.GetRootRegion(), *theta2->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(rvsdg.GetRootRegion(), *theta3->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(rvsdg.GetRootRegion(), *theta4->subregion()));

  // Every region can reach the root region
  EXPECT_TRUE(trace.canRegionReachRegion(*theta0->subregion(), rvsdg.GetRootRegion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta1->subregion(), rvsdg.GetRootRegion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta1->subregion(), rvsdg.GetRootRegion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta1->subregion(), rvsdg.GetRootRegion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta1->subregion(), rvsdg.GetRootRegion()));

  // theta0 can reach every region inside it
  EXPECT_TRUE(trace.canRegionReachRegion(*theta0->subregion(), *theta1->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta0->subregion(), *theta2->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta0->subregion(), *theta3->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta0->subregion(), *theta4->subregion()));

  // theta0 can also be reached by every region inside it
  EXPECT_TRUE(trace.canRegionReachRegion(*theta1->subregion(), *theta0->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta2->subregion(), *theta0->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta3->subregion(), *theta0->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta4->subregion(), *theta0->subregion()));

  // theta2 can be reached from theta1
  EXPECT_TRUE(trace.canRegionReachRegion(*theta1->subregion(), *theta2->subregion()));

  // theta3 and theta4 can reach each other
  EXPECT_TRUE(trace.canRegionReachRegion(*theta3->subregion(), *theta4->subregion()));
  EXPECT_TRUE(trace.canRegionReachRegion(*theta4->subregion(), *theta3->subregion()));
}

TEST(RegionPredicateTraceTests, TraceIntoGamma)
{
  /**
   * Creates an RVSDG graph that looks like:
   *
   * CTRL(0)
   *   v
   * +-gamma0------------------+---------+
   * |                         |         |
   * | CTRL(0)   CTRL(1)       |         |
   * |   v  v      v           |         |
   * | +-gamma1--+---------+   |         |
   * | |   \     |     /   |   |         |
   * | |    \    |    /    |   |         |
   * | |    v    |    v    |   |         |
   * | +---------+---------+   | CTRL(1) |
   * |           v             |    v    |
   * +-------------------------+---------+
   *                    v
   *                +-gamma2---+---------+
   *                |          |         |
   *                +----------+---------+
   *
   * Checks that the regions of gamma2 can only be reached from regions
   * that provide the correct predicate value
   */

  using namespace jlm;

  auto controlType = rvsdg::ControlType::Create(2);

  rvsdg::Graph rvsdg;
  auto & outerCtrl0 = rvsdg::ControlConstantOperation::createFalse(rvsdg.GetRootRegion());
  auto & gamma0 = *rvsdg::GammaNode::create(&outerCtrl0, 2);

  // Left subregion of gamma0
  auto & leftCtrl0 = rvsdg::ControlConstantOperation::createFalse(*gamma0.subregion(0));
  auto & leftCtrl1 = rvsdg::ControlConstantOperation::createTrue(*gamma0.subregion(0));

  auto & gamma1 = *rvsdg::GammaNode::create(&leftCtrl0, 2);
  auto gamma1Entry0 = gamma1.AddEntryVar(&leftCtrl0);
  auto gamma1Entry1 = gamma1.AddEntryVar(&leftCtrl1);
  auto gamma1Exit =
      gamma1.AddExitVar({ gamma1Entry0.branchArgument[0], gamma1Entry1.branchArgument[1] });

  // Right subregion of gamma1
  auto & rightCtrl1 = rvsdg::ControlConstantOperation::createTrue(*gamma0.subregion(1));

  auto gamma0Exit = gamma0.AddExitVar({ gamma1Exit.output, &rightCtrl1 });

  auto & gamma2 = *rvsdg::GammaNode::create(gamma0Exit.output, 2);

  // Assert
  rvsdg::AlternativeRegionPredicateTracer trace;

  // targeting gamma2's left subregion
  ASSERT_TRUE(trace.canRegionReachRegion(*gamma1.subregion(0), *gamma2.subregion(0)));
  ASSERT_FALSE(trace.canRegionReachRegion(*gamma1.subregion(1), *gamma2.subregion(0)));
  ASSERT_TRUE(trace.canRegionReachRegion(*gamma0.subregion(0), *gamma2.subregion(0)));
  ASSERT_FALSE(trace.canRegionReachRegion(*gamma0.subregion(1), *gamma2.subregion(0)));

  // targeting gamma2's right subregion
  ASSERT_FALSE(trace.canRegionReachRegion(*gamma1.subregion(0), *gamma2.subregion(1)));
  ASSERT_TRUE(trace.canRegionReachRegion(*gamma1.subregion(1), *gamma2.subregion(1)));
  ASSERT_TRUE(trace.canRegionReachRegion(*gamma0.subregion(0), *gamma2.subregion(1)));
  ASSERT_TRUE(trace.canRegionReachRegion(*gamma0.subregion(1), *gamma2.subregion(1)));
}

TEST(RegionPredicateTraceTests, TraceThroughGammas)
{
  /**
   * Creates an RVSDG that looks like
   *
   *  TestOp(CtrlType)
   *    v
   * +-gamma0-------+--------------+
   * | Ctrl(0)      | Ctrl(1)      |
   * |   v          |   v          |
   * +---x----------+---x----------+
   *               |
   *               |
   *  TestOp(Ctrl) |  Ctrl(1)
   *    v          v    v
   * +-gamma1-------+--------------+
   * |   \          |         /    |
   * |    \         |        /     |
   * |     \        |       /      |
   * |      v       |      v       |
   * +------x-------+------x-------+
   *     |
   *     v
   * +-gamma2---+---------+---------+
   * |          |         |         |
   * +----------+---------+---------+
   */

  using namespace jlm;

  auto controlType2 = rvsdg::ControlType::Create(2);
  auto controlType3 = rvsdg::ControlType::Create(3);

  rvsdg::Graph rvsdg;

  // gamma 0
  auto & testOp0 =
      rvsdg::CreateOpNode<rvsdg::TestNullaryOperation>(rvsdg.GetRootRegion(), controlType2);
  auto & gamma0 = *rvsdg::GammaNode::create(testOp0.output(0), 2);
  auto & gamma0Ctrl0 = rvsdg::ControlConstantOperation::create(*gamma0.subregion(0), { 0, 3 });
  auto & gamma0Ctrl1 = rvsdg::ControlConstantOperation::create(*gamma0.subregion(1), { 1, 3 });
  auto gamma0ExitVar = gamma0.AddExitVar({ &gamma0Ctrl0, &gamma0Ctrl1 });

  // gamma 1
  auto & testOp1 =
      rvsdg::CreateOpNode<rvsdg::TestNullaryOperation>(rvsdg.GetRootRegion(), controlType2);
  auto & gamma1 = *rvsdg::GammaNode::create(testOp1.output(0), 2);
  auto gamma1EntryFromGamma0 = gamma1.AddEntryVar(gamma0ExitVar.output);
  auto & gamma1Ctrl1 = rvsdg::ControlConstantOperation::create(rvsdg.GetRootRegion(), { 1, 3 });
  auto gamma1EntryFromCtrl1 = gamma1.AddEntryVar(&gamma1Ctrl1);
  auto gamma1ExitVar = gamma1.AddExitVar(
      { gamma1EntryFromGamma0.branchArgument[0], gamma1EntryFromCtrl1.branchArgument[1] });

  // gamma 2
  auto & gamma2 = *rvsdg::GammaNode::create(gamma1ExitVar.output, 3);

  // Assert
  rvsdg::AlternativeRegionPredicateTracer tracer;

  // gamma0 has no effect on the subregions of gamma1
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma0.subregion(0), *gamma1.subregion(0)));
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma0.subregion(0), *gamma1.subregion(1)));
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma0.subregion(1), *gamma1.subregion(0)));
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma0.subregion(1), *gamma1.subregion(1)));

  // gamma0 has no effect on the choice between region 0 or 1 in gamma2 either
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma0.subregion(0), *gamma2.subregion(0)));
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma0.subregion(0), *gamma2.subregion(1)));
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma0.subregion(1), *gamma2.subregion(0)));
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma0.subregion(1), *gamma2.subregion(1)));

  // From subregion 0 of gamma1 both subregions 0 and 1 can be reached in gamma2
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma1.subregion(0), *gamma2.subregion(0)));
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma1.subregion(0), *gamma2.subregion(1)));
  // From subregion 1 of gamma1, however, only subregions 1 can be reached in gamma2
  ASSERT_FALSE(tracer.canRegionReachRegion(*gamma1.subregion(1), *gamma2.subregion(0)));
  ASSERT_TRUE(tracer.canRegionReachRegion(*gamma1.subregion(1), *gamma2.subregion(1)));

  // region 2 of gamma2 is entriely unreachable, from any region, including the root
  ASSERT_FALSE(tracer.canRegionReachRegion(*gamma0.subregion(0), *gamma2.subregion(2)));
  ASSERT_FALSE(tracer.canRegionReachRegion(*gamma0.subregion(1), *gamma2.subregion(2)));
  ASSERT_FALSE(tracer.canRegionReachRegion(*gamma1.subregion(0), *gamma2.subregion(2)));
  ASSERT_FALSE(tracer.canRegionReachRegion(*gamma1.subregion(1), *gamma2.subregion(2)));
  ASSERT_FALSE(tracer.canRegionReachRegion(rvsdg.GetRootRegion(), *gamma2.subregion(2)));

  // Using the old predicate tracer, the following assert fails
  // rvsdg::RegionPredicateTrace oldTracer;
  // ASSERT_TRUE(oldTracer.CheckPredicatesSatisfiable(*gamma0.subregion(0), *gamma2.subregion(1)));
}
