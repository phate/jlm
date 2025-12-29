/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

static bool
region_contains_node(const jlm::rvsdg::Region * region, const jlm::rvsdg::Node * n)
{
  for (const auto & node : region->Nodes())
  {
    if (&node == n)
      return true;
  }

  return false;
}

TEST(GraphTests, test_recursive_prune)
{
  using namespace jlm::rvsdg;

  auto t = TestType::createValueType();

  Graph graph;
  auto & imp = jlm::rvsdg::GraphImport::Create(graph, t, "i");

  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), { &imp }, { t });
  auto n2 = TestOperation::createNode(&graph.GetRootRegion(), { &imp }, { t });

  auto n3 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto input0 = n3->addInputWithArguments(imp);
  auto & a1 = *n3->addArguments(t).argument[0];
  auto n4 = TestOperation::createNode(n3->subregion(0), { &a1 }, { t });
  auto n5 = TestOperation::createNode(n3->subregion(0), { &a1 }, { t });
  auto o1 = n3->addOutputWithResults({ n4->output(0) });

  auto n6 = TestStructuralNode::create(n3->subregion(0), 1);

  GraphExport::Create(*n2->output(0), "n2");
  GraphExport::Create(*o1.output, "n3");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);
  graph.PruneNodes();
  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  EXPECT_FALSE(region_contains_node(&graph.GetRootRegion(), n1));
  EXPECT_TRUE(region_contains_node(&graph.GetRootRegion(), n2));
  EXPECT_TRUE(region_contains_node(&graph.GetRootRegion(), n3));
  EXPECT_TRUE(region_contains_node(n3->subregion(0), n4));
  EXPECT_FALSE(region_contains_node(n3->subregion(0), n5));
  EXPECT_FALSE(region_contains_node(n3->subregion(0), n6));
}

TEST(GraphTests, test_empty_graph_pruning)
{
  jlm::rvsdg::Graph graph;

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  graph.PruneNodes();

  EXPECT_EQ(graph.GetRootRegion().numNodes(), 0);

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);
}

TEST(GraphTests, test_prune_replace)
{
  using namespace jlm::rvsdg;

  auto type = TestType::createValueType();

  Graph graph;
  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), {}, { type });
  auto n2 = TestOperation::createNode(&graph.GetRootRegion(), { n1->output(0) }, { type });
  auto n3 = TestOperation::createNode(&graph.GetRootRegion(), { n2->output(0) }, { type });

  GraphExport::Create(*n2->output(0), "n2");
  GraphExport::Create(*n3->output(0), "n3");

  auto n4 = TestOperation::createNode(&graph.GetRootRegion(), { n1->output(0) }, { type });

  n2->output(0)->divert_users(n4->output(0));
  EXPECT_EQ(n2->output(0)->nusers(), 0);

  graph.PruneNodes();

  EXPECT_FALSE(region_contains_node(&graph.GetRootRegion(), n2));
}

TEST(GraphTests, Copy)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto & argument = jlm::rvsdg::GraphImport::Create(graph, valueType, "import");
  auto node = TestOperation::createNode(&graph.GetRootRegion(), { &argument }, { valueType });
  GraphExport::Create(*node->output(0), "export");

  // Act
  auto newGraph = graph.Copy();

  // Assert
  EXPECT_EQ(newGraph->GetRootRegion().narguments(), 1);
  auto copiedArgument = newGraph->GetRootRegion().argument(0);
  EXPECT_TRUE(is<jlm::rvsdg::GraphImport>(copiedArgument));

  EXPECT_EQ(newGraph->GetRootRegion().numNodes(), 1);
  auto copiedNode = newGraph->GetRootRegion().Nodes().begin().ptr();
  EXPECT_EQ(copiedNode->ninputs() == 1 && copiedNode->noutputs(), 1);
  EXPECT_EQ(copiedNode->input(0)->origin(), copiedArgument);

  EXPECT_EQ(newGraph->GetRootRegion().nresults(), 1);
  auto copiedResult = newGraph->GetRootRegion().result(0);
  EXPECT_TRUE(is<jlm::rvsdg::GraphExport>(*copiedResult));
  EXPECT_EQ(copiedResult->origin(), copiedNode->output(0));
}
