/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/traverser.hpp>

TEST(BottomUpTraverserTests, testInitialization)
{
  using namespace jlm::rvsdg;

  jlm::rvsdg::Graph graph;
  auto vtype = jlm::rvsdg::TestType::createValueType();
  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), {}, {});
  auto n2 = TestOperation::createNode(&graph.GetRootRegion(), {}, { vtype });

  jlm::rvsdg::GraphExport::Create(*n2->output(0), "dummy");

  bool n1_visited = false;
  bool n2_visited = false;
  for (const auto & node : jlm::rvsdg::BottomUpTraverser(&graph.GetRootRegion()))
  {
    if (node == n1)
      n1_visited = true;
    if (node == n2)
      n2_visited = true;
  }

  EXPECT_TRUE(n1_visited);
  EXPECT_TRUE(n2_visited);
}

TEST(BottomUpTraverserTests, testBasicTraversal)
{
  using namespace jlm::rvsdg;

  jlm::rvsdg::Graph graph;
  auto type = jlm::rvsdg::TestType::createValueType();
  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), {}, { type, type });
  auto n2 =
      TestOperation::createNode(&graph.GetRootRegion(), { n1->output(0), n1->output(1) }, { type });

  jlm::rvsdg::GraphExport::Create(*n2->output(0), "dummy");

  {
    const jlm::rvsdg::Node * tmp = nullptr;
    jlm::rvsdg::BottomUpTraverser trav(&graph.GetRootRegion());
    tmp = trav.next();
    EXPECT_EQ(tmp, n2);
    tmp = trav.next();
    EXPECT_EQ(tmp, n1);
    tmp = trav.next();
    EXPECT_EQ(tmp, nullptr);
  }
}

TEST(BottomUpTraverserTests, testOrderEnforcement)
{
  using namespace jlm::rvsdg;

  jlm::rvsdg::Graph graph;
  auto type = jlm::rvsdg::TestType::createValueType();
  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), {}, { type, type });
  auto n2 = TestOperation::createNode(&graph.GetRootRegion(), { n1->output(0) }, { type });
  auto n3 =
      TestOperation::createNode(&graph.GetRootRegion(), { n2->output(0), n1->output(1) }, { type });

  {
    const jlm::rvsdg::Node * tmp = nullptr;
    jlm::rvsdg::BottomUpTraverser trav(&graph.GetRootRegion());

    tmp = trav.next();
    EXPECT_EQ(tmp, n3);
    tmp = trav.next();
    EXPECT_EQ(tmp, n2);
    tmp = trav.next();
    EXPECT_EQ(tmp, n1);
    tmp = trav.next();
    EXPECT_EQ(tmp, nullptr);
  }
}
