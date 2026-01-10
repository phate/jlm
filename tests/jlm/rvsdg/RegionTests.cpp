/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/util/AnnotationMap.hpp>

#include <algorithm>
#include <cassert>

TEST(RegionTests, IteratorRanges)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();

  jlm::rvsdg::Graph graph;

  auto structuralNode = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto & subregion = *structuralNode->subregion(0);
  auto & constSubregion = *static_cast<const jlm::rvsdg::Region *>(structuralNode->subregion(0));

  auto & argument0 = *structuralNode->addArguments(valueType).argument[0];
  auto & argument1 = *structuralNode->addArguments(valueType).argument[0];

  auto topNode0 = TestOperation::createNode(&subregion, {}, { valueType });
  auto node0 = TestOperation::createNode(&subregion, { &argument0 }, { valueType });
  auto node1 = TestOperation::createNode(&subregion, { &argument1 }, { valueType });
  auto bottomNode0 =
      TestOperation::createNode(&subregion, { &argument0, &argument1 }, { valueType });

  const auto outputVar0 = structuralNode->addResults({ topNode0->output(0) });
  const auto outputVar1 = structuralNode->addResults({ node0->output(0) });
  const auto outputVar2 = structuralNode->addResults({ node1->output(0) });

  // Act & Assert
  auto numArguments = std::distance(subregion.Arguments().begin(), subregion.Arguments().end());
  EXPECT_EQ(numArguments, 2);
  for (auto & argument : constSubregion.Arguments())
  {
    EXPECT_TRUE(argument == &argument0 || argument == &argument1);
  }

  auto numTopNodes = std::distance(subregion.TopNodes().begin(), subregion.TopNodes().end());
  EXPECT_EQ(numTopNodes, 1);
  for (auto & topNode : constSubregion.TopNodes())
  {
    EXPECT_EQ(&topNode, topNode0);
  }

  auto numNodes = std::distance(subregion.Nodes().begin(), subregion.Nodes().end());
  EXPECT_EQ(numNodes, 4);
  for (auto & node : constSubregion.Nodes())
  {
    EXPECT_TRUE(&node == topNode0 || &node == node0 || &node == node1 || &node == bottomNode0);
  }

  auto numBottomNodes =
      std::distance(subregion.BottomNodes().begin(), subregion.BottomNodes().end());
  EXPECT_EQ(numBottomNodes, 1);
  for (auto & bottomNode : constSubregion.BottomNodes())
  {
    EXPECT_EQ(&bottomNode, bottomNode0);
  }

  auto numResults = std::distance(subregion.Results().begin(), subregion.Results().end());
  EXPECT_EQ(numResults, 3);
  for (auto & result : constSubregion.Results())
  {
    EXPECT_TRUE(
        result == outputVar0.result[0] || result == outputVar1.result[0]
        || result == outputVar2.result[0]);
  }
}

TEST(RegionTests, Contains)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();

  jlm::rvsdg::Graph graph;
  auto import = &jlm::rvsdg::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto inputVar1 = structuralNode1->addInputWithArguments(*import);
  TestUnaryOperation::create(
      structuralNode1->subregion(0),
      valueType,
      inputVar1.argument[0],
      valueType);

  auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto inputVar2 = structuralNode2->addInputWithArguments(*import);
  TestBinaryOperation::create(valueType, valueType, inputVar2.argument[0], inputVar2.argument[0]);

  // Act & Assert
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsNodeType<TestStructuralNode>(graph.GetRootRegion(), false));
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<TestUnaryOperation>(graph.GetRootRegion(), true));
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<TestBinaryOperation>(graph.GetRootRegion(), true));
  EXPECT_TRUE(!jlm::rvsdg::Region::ContainsOperation<TestOperation>(graph.GetRootRegion(), true));
}

TEST(RegionTests, IsRootRegion)
{
  using namespace jlm::rvsdg;

  // Arrange
  jlm::rvsdg::Graph graph;

  auto structuralNode = TestStructuralNode::create(&graph.GetRootRegion(), 1);

  // Act & Assert
  EXPECT_TRUE(graph.GetRootRegion().IsRootRegion());
  EXPECT_FALSE(structuralNode->subregion(0)->IsRootRegion());
}

TEST(RegionTests, NumRegions_EmptyRvsdg)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  // Act & Assert
  EXPECT_EQ(Region::NumRegions(graph.GetRootRegion()), 1u);
}

TEST(RegionTests, NumRegions_NonEmptyRvsdg)
{
  using namespace jlm::rvsdg;

  // Arrange
  const Graph graph;
  auto structuralNode = TestStructuralNode::create(&graph.GetRootRegion(), 4);
  TestStructuralNode::create(structuralNode->subregion(0), 2);
  TestStructuralNode::create(structuralNode->subregion(3), 5);

  // Act & Assert
  constexpr unsigned int numTotalSubRegions = 1 + 4 + 2 + 5;
  EXPECT_EQ(Region::NumRegions(graph.GetRootRegion()), numTotalSubRegions);
}

TEST(RegionTests, RemoveResults)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  const RecordingObserver observer(rootRegion);

  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i2 = GraphImport::Create(rvsdg, valueType, "i2");
  auto & i3 = GraphImport::Create(rvsdg, valueType, "i3");
  auto & i4 = GraphImport::Create(rvsdg, valueType, "i4");
  auto & i5 = GraphImport::Create(rvsdg, valueType, "i5");
  auto & i6 = GraphImport::Create(rvsdg, valueType, "i6");
  auto & i7 = GraphImport::Create(rvsdg, valueType, "i7");
  auto & i8 = GraphImport::Create(rvsdg, valueType, "i8");
  auto & i9 = GraphImport::Create(rvsdg, valueType, "i9");

  GraphExport::Create(i0, "x0");
  GraphExport::Create(i1, "x1");
  GraphExport::Create(i2, "x2");
  GraphExport::Create(i3, "x3");
  GraphExport::Create(i4, "x4");
  GraphExport::Create(i5, "x5");
  GraphExport::Create(i6, "x6");
  GraphExport::Create(i7, "x7");
  GraphExport::Create(i8, "x8");
  GraphExport::Create(i9, "x9");

  // Act & Arrange
  EXPECT_EQ(rvsdg.GetRootRegion().nresults(), 10u);

  // Remove all results that have an even index
  size_t numRemovedResults = rootRegion.RemoveResults({ 0, 2, 4, 6, 8 });
  EXPECT_EQ(numRemovedResults, 5u);
  EXPECT_EQ(rootRegion.nresults(), 5u);
  EXPECT_EQ(rootRegion.result(0)->origin(), &i1);
  EXPECT_EQ(rootRegion.result(1)->origin(), &i3);
  EXPECT_EQ(rootRegion.result(2)->origin(), &i5);
  EXPECT_EQ(rootRegion.result(3)->origin(), &i7);
  EXPECT_EQ(rootRegion.result(4)->origin(), &i9);
  EXPECT_EQ(i0.nusers(), 0u);
  EXPECT_EQ(i2.nusers(), 0u);
  EXPECT_EQ(i4.nusers(), 0u);
  EXPECT_EQ(i6.nusers(), 0u);
  EXPECT_EQ(i8.nusers(), 0u);
  EXPECT_EQ(observer.destroyedInputIndices(), std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove no result
  numRemovedResults = rootRegion.RemoveResults({});
  EXPECT_EQ(numRemovedResults, 0u);
  EXPECT_EQ(rootRegion.nresults(), 5u);
  EXPECT_EQ(observer.destroyedInputIndices(), std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove non-existent input
  numRemovedResults = rootRegion.RemoveResults({ 15 });
  EXPECT_EQ(numRemovedResults, 0u);
  EXPECT_EQ(rootRegion.nresults(), 5u);
  EXPECT_EQ(observer.destroyedInputIndices(), std::vector<size_t>({ 0, 2, 4, 6, 8 }));

  // Remove remaining results
  numRemovedResults = rootRegion.RemoveResults({ 0, 1, 2, 3, 4 });
  EXPECT_EQ(numRemovedResults, 5u);
  EXPECT_EQ(rootRegion.nresults(), 0u);
  EXPECT_EQ(i1.nusers(), 0u);
  EXPECT_EQ(i3.nusers(), 0u);
  EXPECT_EQ(i5.nusers(), 0u);
  EXPECT_EQ(i7.nusers(), 0u);
  EXPECT_EQ(i9.nusers(), 0u);
  EXPECT_EQ(
      observer.destroyedInputIndices(),
      std::vector<size_t>({ 0, 2, 4, 6, 8, 0, 1, 2, 3, 4 }));
}

TEST(RegionTests, RemoveArguments)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();

  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto argument0 = &GraphImport::Create(rvsdg, valueType, "argument0");
  auto argument1 = &GraphImport::Create(rvsdg, valueType, "argument1");
  auto argument2 = &GraphImport::Create(rvsdg, valueType, "argument2");
  auto argument3 = &GraphImport::Create(rvsdg, valueType, "argument3");
  auto argument4 = &GraphImport::Create(rvsdg, valueType, "argument4");
  auto argument5 = &GraphImport::Create(rvsdg, valueType, "argument5");
  auto argument6 = &GraphImport::Create(rvsdg, valueType, "argument6");
  auto argument7 = &GraphImport::Create(rvsdg, valueType, "argument7");
  auto argument8 = &GraphImport::Create(rvsdg, valueType, "argument8");
  auto argument9 = &GraphImport::Create(rvsdg, valueType, "argument9");

  auto node = TestOperation::createNode(
      &rvsdg.GetRootRegion(),
      { argument2, argument4, argument6 },
      { valueType });

  // Act & Arrange
  EXPECT_EQ(rootRegion.narguments(), 10u);
  EXPECT_EQ(argument0->index(), 0u);
  EXPECT_EQ(argument1->index(), 1u);
  EXPECT_EQ(argument2->index(), 2u);
  EXPECT_EQ(argument3->index(), 3u);
  EXPECT_EQ(argument4->index(), 4u);
  EXPECT_EQ(argument5->index(), 5u);
  EXPECT_EQ(argument6->index(), 6u);
  EXPECT_EQ(argument7->index(), 7u);
  EXPECT_EQ(argument8->index(), 8u);
  EXPECT_EQ(argument9->index(), 9u);

  // Remove all arguments that have an even index
  size_t numRemovedArguments = rootRegion.RemoveArguments({ 0, 2, 4, 6, 8 });
  // We expect only argument0 and argument8 to be removed, as argument2, argument4, and
  // argument6 are not dead
  EXPECT_EQ(numRemovedArguments, 2u);
  EXPECT_EQ(rootRegion.narguments(), 8u);
  EXPECT_EQ(argument1->index(), 0u);
  EXPECT_EQ(argument2->index(), 1u);
  EXPECT_EQ(argument3->index(), 2u);
  EXPECT_EQ(argument4->index(), 3u);
  EXPECT_EQ(argument5->index(), 4u);
  EXPECT_EQ(argument6->index(), 5u);
  EXPECT_EQ(argument7->index(), 6u);
  EXPECT_EQ(argument9->index(), 7u);

  // Reassign arguments to avoid mental gymnastics
  argument0 = argument1;
  argument1 = argument2;
  argument2 = argument3;
  argument3 = argument4;
  argument4 = argument5;
  argument5 = argument6;
  argument6 = argument7;
  argument7 = argument9;

  // Remove all users from the arguments
  rootRegion.removeNode(node);

  // Remove all arguments that have an even index
  numRemovedArguments = rootRegion.RemoveArguments({ 0, 2, 4, 6 });
  // We expect argument0, argument2, argument4, and argument6 to be removed
  EXPECT_EQ(numRemovedArguments, 4u);
  EXPECT_EQ(rootRegion.narguments(), 4u);
  EXPECT_EQ(argument1->index(), 0u);
  EXPECT_EQ(argument3->index(), 1u);
  EXPECT_EQ(argument5->index(), 2u);
  EXPECT_EQ(argument7->index(), 3u);

  // Reassign arguments to avoid mental gymnastics
  argument0 = argument1;
  argument1 = argument3;
  argument2 = argument5;
  argument3 = argument7;

  // Remove no argument
  numRemovedArguments = rootRegion.RemoveArguments({});
  EXPECT_EQ(numRemovedArguments, 0u);
  EXPECT_EQ(rootRegion.narguments(), 4u);
  EXPECT_EQ(argument0->index(), 0u);
  EXPECT_EQ(argument1->index(), 1u);
  EXPECT_EQ(argument2->index(), 2u);
  EXPECT_EQ(argument3->index(), 3u);

  // Remove non-existent argument
  numRemovedArguments = rootRegion.RemoveArguments({ 15 });
  EXPECT_EQ(numRemovedArguments, 0u);
  EXPECT_EQ(rootRegion.narguments(), 4u);

  // Remove all remaining arguments
  numRemovedArguments = rootRegion.RemoveArguments({ 0, 1, 2, 3 });
  EXPECT_EQ(numRemovedArguments, 4u);
  EXPECT_EQ(rootRegion.narguments(), 0u);
}

TEST(RegionTests, PruneArguments)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 1);

  auto & argument0 = *structuralNode->addArguments(valueType).argument[0];
  structuralNode->addArguments(valueType);
  auto & argument2 = *structuralNode->addArguments(valueType).argument[0];

  auto node = TestOperation::createNode(
      structuralNode->subregion(0),
      { &argument0, &argument2 },
      { valueType });

  // Act & Arrange
  EXPECT_EQ(structuralNode->subregion(0)->narguments(), 3u);

  size_t numRemovedArguments = structuralNode->subregion(0)->PruneArguments();
  EXPECT_EQ(numRemovedArguments, 1u);
  EXPECT_EQ(structuralNode->subregion(0)->narguments(), 2u);
  EXPECT_EQ(argument0.index(), 0u);
  EXPECT_EQ(argument2.index(), 1u);

  structuralNode->subregion(0)->removeNode(node);
  numRemovedArguments = structuralNode->subregion(0)->PruneArguments();
  EXPECT_EQ(numRemovedArguments, 2u);
  EXPECT_EQ(structuralNode->subregion(0)->narguments(), 0u);
}

TEST(RegionTests, ToTree_EmptyRvsdg)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;

  // Act
  auto tree = Region::ToTree(rvsdg.GetRootRegion());
  std::cout << tree << std::flush;

  // Assert
  EXPECT_EQ(tree, "RootRegion\n");
}

TEST(RegionTests, ToTree_EmptyRvsdgWithAnnotations)
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  Graph rvsdg;

  AnnotationMap annotationMap;
  annotationMap.AddAnnotation(
      &rvsdg.GetRootRegion(),
      Annotation("NumNodes", static_cast<uint64_t>(rvsdg.GetRootRegion().numNodes())));

  // Act
  auto tree = Region::ToTree(rvsdg.GetRootRegion(), annotationMap);
  std::cout << tree << std::flush;

  // Assert
  EXPECT_EQ(tree, "RootRegion NumNodes:0\n");
}

TEST(RegionTests, ToTree_RvsdgWithStructuralNodes)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 2);
  TestStructuralNode::create(structuralNode->subregion(0), 1);
  TestStructuralNode::create(structuralNode->subregion(1), 3);

  // Act
  auto tree = Region::ToTree(rvsdg.GetRootRegion());
  std::cout << tree << std::flush;

  // Assert
  auto numLines = std::count(tree.begin(), tree.end(), '\n');

  // We should find '\n' 10 times: 1 root region + 3 structural nodes + 6 subregions
  EXPECT_EQ(numLines, 10);

  // Check that the last line printed looks accordingly
  auto lastLine = std::string("----Region[2]\n");
  EXPECT_EQ(tree.compare(tree.size() - lastLine.size(), lastLine.size(), lastLine), 0);
}

TEST(RegionTests, ToTree_RvsdgWithStructuralNodesAndAnnotations)
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  Graph rvsdg;
  auto structuralNode1 = TestStructuralNode::create(&rvsdg.GetRootRegion(), 2);
  auto structuralNode2 = TestStructuralNode::create(structuralNode1->subregion(1), 3);
  auto subregion2 = structuralNode2->subregion(2);

  AnnotationMap annotationMap;
  annotationMap.AddAnnotation(
      subregion2,
      Annotation("NumNodes", static_cast<uint64_t>(subregion2->numNodes())));
  annotationMap.AddAnnotation(
      subregion2,
      Annotation("NumArguments", static_cast<uint64_t>(subregion2->narguments())));

  // Act
  auto tree = Region::ToTree(rvsdg.GetRootRegion(), annotationMap);
  std::cout << tree << std::flush;

  // Assert
  auto numLines = std::count(tree.begin(), tree.end(), '\n');

  // We should find '\n' 8 times: 1 root region + 2 structural nodes + 5 subregions
  EXPECT_EQ(numLines, 8);

  // Check that the last line printed looks accordingly
  auto lastLine = std::string("----Region[2] NumNodes:0 NumArguments:0\n");
  EXPECT_EQ(tree.compare(tree.size() - lastLine.size(), lastLine.size(), lastLine), 0);
}

TEST(RegionTests, BottomNodeTests)
{
  using namespace jlm::rvsdg;

  auto valueType = TestType::createValueType();

  // Arrange
  Graph rvsdg;

  // Act & Assert
  // A newly created node without any users should automatically be added to the bottom nodes
  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 1);
  EXPECT_TRUE(structuralNode->IsDead());
  EXPECT_EQ(rvsdg.GetRootRegion().numBottomNodes(), 1u);
  EXPECT_EQ(&*(rvsdg.GetRootRegion().BottomNodes().begin()), structuralNode);

  // The node cedes to be dead
  auto & output = structuralNode->addOutputOnly(valueType);
  GraphExport::Create(output, "x");
  EXPECT_FALSE(structuralNode->IsDead());
  EXPECT_EQ(rvsdg.GetRootRegion().numBottomNodes(), 0u);
  EXPECT_EQ(rvsdg.GetRootRegion().BottomNodes().begin(), rvsdg.GetRootRegion().BottomNodes().end());

  // And it becomes dead again
  rvsdg.GetRootRegion().RemoveResults({ 0 });
  EXPECT_TRUE(structuralNode->IsDead());
  EXPECT_EQ(rvsdg.GetRootRegion().numBottomNodes(), 1u);
  EXPECT_EQ(&*(rvsdg.GetRootRegion().BottomNodes().begin()), structuralNode);
}

TEST(RegionTests, computeDepthMap)
{
  // Arrange
  using namespace jlm::rvsdg;

  auto valueType = TestType::createValueType();

  Graph rvsdg;

  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");

  auto node0 = TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { valueType });
  auto node1 =
      TestOperation::createNode(&rvsdg.GetRootRegion(), { node0->output(0), &i0 }, { valueType });
  auto node2 = TestOperation::createNode(&rvsdg.GetRootRegion(), { &i1 }, { valueType });
  auto node3 = TestOperation::createNode(
      &rvsdg.GetRootRegion(),
      { node1->output(0), node2->output(0) },
      { valueType });

  GraphExport::Create(*node3->output(0), "x0");

  // Act
  const auto depthMap = computeDepthMap(rvsdg.GetRootRegion());

  // Assert
  EXPECT_EQ(depthMap.size(), 4u);
  EXPECT_EQ(depthMap.at(node0), 0u);
  EXPECT_EQ(depthMap.at(node1), 1u);
  EXPECT_EQ(depthMap.at(node2), 0u);
  EXPECT_EQ(depthMap.at(node3), 2u);
}
