/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

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
