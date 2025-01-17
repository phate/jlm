/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/util/AnnotationMap.hpp>

#include <algorithm>
#include <cassert>

static int
IteratorRanges()
{
  using namespace jlm::tests;

  // Arrange
  auto valueType = valuetype::Create();

  jlm::rvsdg::Graph graph;

  auto structuralNode = structural_node::create(&graph.GetRootRegion(), 1);
  auto & subregion = *structuralNode->subregion(0);
  auto & constSubregion = *static_cast<const jlm::rvsdg::Region *>(structuralNode->subregion(0));

  auto & argument0 = TestGraphArgument::Create(subregion, nullptr, valueType);
  auto & argument1 = TestGraphArgument::Create(subregion, nullptr, valueType);

  auto topNode0 = test_op::create(&subregion, {}, { valueType });
  auto node0 = test_op::create(&subregion, { &argument0 }, { valueType });
  auto node1 = test_op::create(&subregion, { &argument1 }, { valueType });
  auto bottomNode0 = test_op::create(&subregion, { &argument0, &argument1 }, { valueType });

  auto & result0 = TestGraphResult::Create(*topNode0->output(0), nullptr);
  auto & result1 = TestGraphResult::Create(*node0->output(0), nullptr);
  auto & result2 = TestGraphResult::Create(*node1->output(0), nullptr);

  // Act & Assert
  auto numArguments = std::distance(subregion.Arguments().begin(), subregion.Arguments().end());
  assert(numArguments == 2);
  for (auto & argument : constSubregion.Arguments())
  {
    assert(argument == &argument0 || argument == &argument1);
  }

  auto numTopNodes = std::distance(subregion.TopNodes().begin(), subregion.TopNodes().end());
  assert(numTopNodes == 1);
  for (auto & topNode : constSubregion.TopNodes())
  {
    assert(&topNode == topNode0);
  }

  auto numNodes = std::distance(subregion.Nodes().begin(), subregion.Nodes().end());
  assert(numNodes == 4);
  for (auto & node : constSubregion.Nodes())
  {
    assert(&node == topNode0 || &node == node0 || &node == node1 || &node == bottomNode0);
  }

  auto numBottomNodes =
      std::distance(subregion.BottomNodes().begin(), subregion.BottomNodes().end());
  assert(numBottomNodes == 1);
  for (auto & bottomNode : constSubregion.BottomNodes())
  {
    assert(&bottomNode == bottomNode0);
  }

  auto numResults = std::distance(subregion.Results().begin(), subregion.Results().end());
  assert(numResults == 3);
  for (auto & result : constSubregion.Results())
  {
    assert(result == &result0 || result == &result1 || result == &result2);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-IteratorRanges", IteratorRanges)

/**
 * Test Region::Contains().
 */
static int
Contains()
{
  using namespace jlm::tests;

  // Arrange
  auto valueType = valuetype::Create();

  jlm::rvsdg::Graph graph;
  auto import = &jlm::tests::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = structural_node::create(&graph.GetRootRegion(), 1);
  auto structuralInput1 = jlm::rvsdg::StructuralInput::create(structuralNode1, import, valueType);
  auto & regionArgument1 =
      TestGraphArgument::Create(*structuralNode1->subregion(0), structuralInput1, valueType);
  unary_op::create(structuralNode1->subregion(0), valueType, &regionArgument1, valueType);

  auto structuralNode2 = structural_node::create(&graph.GetRootRegion(), 1);
  auto structuralInput2 = jlm::rvsdg::StructuralInput::create(structuralNode2, import, valueType);
  auto & regionArgument2 =
      TestGraphArgument::Create(*structuralNode2->subregion(0), structuralInput2, valueType);
  binary_op::create(valueType, valueType, &regionArgument2, &regionArgument2);

  // Act & Assert
  assert(jlm::rvsdg::Region::Contains<structural_op>(graph.GetRootRegion(), false));
  assert(jlm::rvsdg::Region::Contains<unary_op>(graph.GetRootRegion(), true));
  assert(jlm::rvsdg::Region::Contains<binary_op>(graph.GetRootRegion(), true));
  assert(!jlm::rvsdg::Region::Contains<test_op>(graph.GetRootRegion(), true));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-Contains", Contains)

/**
 * Test Region::IsRootRegion().
 */
static int
IsRootRegion()
{
  // Arrange
  jlm::rvsdg::Graph graph;

  auto structuralNode = jlm::tests::structural_node::create(&graph.GetRootRegion(), 1);

  // Act & Assert
  assert(graph.GetRootRegion().IsRootRegion());
  assert(!structuralNode->subregion(0)->IsRootRegion());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-IsRootRegion", IsRootRegion)

/**
 * Test Region::NumRegions() with an empty Rvsdg.
 */
static int
NumRegions_EmptyRvsdg()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  // Act & Assert
  assert(Region::NumRegions(graph.GetRootRegion()) == 1);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-NumRegions_EmptyRvsdg", NumRegions_EmptyRvsdg)

/**
 * Test Region::NumRegions() with non-empty Rvsdg.
 */
static int
NumRegions_NonEmptyRvsdg()
{
  using namespace jlm::rvsdg;

  // Arrange
  const Graph graph;
  auto structuralNode = jlm::tests::structural_node::create(&graph.GetRootRegion(), 4);
  jlm::tests::structural_node::create(structuralNode->subregion(0), 2);
  jlm::tests::structural_node::create(structuralNode->subregion(3), 5);

  // Act & Assert
  assert(Region::NumRegions(graph.GetRootRegion()) == 1 + 4 + 2 + 5);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-NumRegions_NonEmptyRvsdg", NumRegions_NonEmptyRvsdg)

/**
 * Test Region::RemoveResultsWhere()
 */
static int
RemoveResultsWhere()
{
  using namespace jlm::tests;

  // Arrange
  jlm::rvsdg::Graph rvsdg;
  jlm::rvsdg::Region region(&rvsdg.GetRootRegion(), &rvsdg);

  auto valueType = jlm::tests::valuetype::Create();
  auto node = jlm::tests::test_op::Create(&region, {}, {}, { valueType });

  auto & result0 = TestGraphResult::Create(*node->output(0), nullptr);
  auto & result1 = TestGraphResult::Create(*node->output(0), nullptr);
  auto & result2 = TestGraphResult::Create(*node->output(0), nullptr);

  // Act & Arrange
  assert(region.nresults() == 3);
  assert(result0.index() == 0);
  assert(result1.index() == 1);
  assert(result2.index() == 2);

  region.RemoveResultsWhere(
      [](const jlm::rvsdg::RegionResult & result)
      {
        return result.index() == 1;
      });
  assert(region.nresults() == 2);
  assert(result0.index() == 0);
  assert(result2.index() == 1);

  region.RemoveResultsWhere(
      [](const jlm::rvsdg::RegionResult &)
      {
        return false;
      });
  assert(region.nresults() == 2);
  assert(result0.index() == 0);
  assert(result2.index() == 1);

  region.RemoveResultsWhere(
      [](const jlm::rvsdg::RegionResult &)
      {
        return true;
      });
  assert(region.nresults() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-RemoveResultsWhere", RemoveResultsWhere)

/**
 * Test Region::RemoveArgumentsWhere()
 */
static int
RemoveArgumentsWhere()
{
  using namespace jlm::tests;

  // Arrange
  jlm::rvsdg::Graph rvsdg;
  jlm::rvsdg::Region region(&rvsdg.GetRootRegion(), &rvsdg);

  auto valueType = jlm::tests::valuetype::Create();
  auto & argument0 = TestGraphArgument::Create(region, nullptr, valueType);
  auto & argument1 = TestGraphArgument::Create(region, nullptr, valueType);
  auto & argument2 = TestGraphArgument::Create(region, nullptr, valueType);

  auto node = jlm::tests::test_op::Create(&region, { valueType }, { &argument1 }, { valueType });

  // Act & Arrange
  assert(region.narguments() == 3);
  assert(argument0.index() == 0);
  assert(argument1.index() == 1);
  assert(argument2.index() == 2);

  region.RemoveArgumentsWhere(
      [](const jlm::rvsdg::RegionArgument &)
      {
        return true;
      });
  assert(region.narguments() == 1);
  assert(argument1.index() == 0);

  region.remove_node(node);
  region.RemoveArgumentsWhere(
      [](const jlm::rvsdg::RegionArgument &)
      {
        return false;
      });
  assert(region.narguments() == 1);
  assert(argument1.index() == 0);

  region.RemoveArgumentsWhere(
      [](const jlm::rvsdg::RegionArgument & argument)
      {
        return argument.index() == 0;
      });
  assert(region.narguments() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-RemoveArgumentsWhere", RemoveArgumentsWhere)

/**
 * Test Region::PruneArguments()
 */
static int
PruneArguments()
{
  using namespace jlm::tests;

  // Arrange
  jlm::rvsdg::Graph rvsdg;
  jlm::rvsdg::Region region(&rvsdg.GetRootRegion(), &rvsdg);

  auto valueType = jlm::tests::valuetype::Create();
  auto & argument0 = TestGraphArgument::Create(region, nullptr, valueType);
  TestGraphArgument::Create(region, nullptr, valueType);
  auto & argument2 = TestGraphArgument::Create(region, nullptr, valueType);

  auto node = jlm::tests::test_op::Create(
      &region,
      { valueType, valueType },
      { &argument0, &argument2 },
      { valueType });

  // Act & Arrange
  assert(region.narguments() == 3);

  region.PruneArguments();
  assert(region.narguments() == 2);
  assert(argument0.index() == 0);
  assert(argument2.index() == 1);

  region.remove_node(node);
  region.PruneArguments();
  assert(region.narguments() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-PruneArguments", PruneArguments)

static int
ToTree_EmptyRvsdg()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;

  // Act
  auto tree = Region::ToTree(rvsdg.GetRootRegion());
  std::cout << tree << std::flush;

  // Assert
  assert(tree == "RootRegion\n");

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-ToTree_EmptyRvsdg", ToTree_EmptyRvsdg)

static int
ToTree_EmptyRvsdgWithAnnotations()
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  Graph rvsdg;

  AnnotationMap annotationMap;
  annotationMap.AddAnnotation(
      &rvsdg.GetRootRegion(),
      Annotation("NumNodes", rvsdg.GetRootRegion().nnodes()));

  // Act
  auto tree = Region::ToTree(rvsdg.GetRootRegion(), annotationMap);
  std::cout << tree << std::flush;

  // Assert
  assert(tree == "RootRegion NumNodes:0\n");

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/RegionTests-ToTree_EmptyRvsdgWithAnnotations",
    ToTree_EmptyRvsdgWithAnnotations)

static int
ToTree_RvsdgWithStructuralNodes()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto structuralNode = jlm::tests::structural_node::create(&rvsdg.GetRootRegion(), 2);
  jlm::tests::structural_node::create(structuralNode->subregion(0), 1);
  jlm::tests::structural_node::create(structuralNode->subregion(1), 3);

  // Act
  auto tree = Region::ToTree(rvsdg.GetRootRegion());
  std::cout << tree << std::flush;

  // Assert
  auto numLines = std::count(tree.begin(), tree.end(), '\n');

  // We should find '\n' 10 times: 1 root region + 3 structural nodes + 6 subregions
  assert(numLines == 10);

  // Check that the last line printed looks accordingly
  auto lastLine = std::string("----Region[2]\n");
  assert(tree.compare(tree.size() - lastLine.size(), lastLine.size(), lastLine) == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/RegionTests-ToTree_RvsdgWithStructuralNodes",
    ToTree_RvsdgWithStructuralNodes)

static int
ToTree_RvsdgWithStructuralNodesAndAnnotations()
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  Graph rvsdg;
  auto structuralNode1 = jlm::tests::structural_node::create(&rvsdg.GetRootRegion(), 2);
  auto structuralNode2 = jlm::tests::structural_node::create(structuralNode1->subregion(1), 3);
  auto subregion2 = structuralNode2->subregion(2);

  AnnotationMap annotationMap;
  annotationMap.AddAnnotation(subregion2, Annotation("NumNodes", subregion2->nnodes()));
  annotationMap.AddAnnotation(subregion2, Annotation("NumArguments", subregion2->narguments()));

  // Act
  auto tree = Region::ToTree(rvsdg.GetRootRegion(), annotationMap);
  std::cout << tree << std::flush;

  // Assert
  auto numLines = std::count(tree.begin(), tree.end(), '\n');

  // We should find '\n' 8 times: 1 root region + 2 structural nodes + 5 subregions
  assert(numLines == 8);

  // Check that the last line printed looks accordingly
  auto lastLine = std::string("----Region[2] NumNodes:0 NumArguments:0\n");
  assert(tree.compare(tree.size() - lastLine.size(), lastLine.size(), lastLine) == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/RegionTests-ToTree_RvsdgWithStructuralNodesAndAnnotations",
    ToTree_RvsdgWithStructuralNodesAndAnnotations)

static int
BottomNodeTests()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto valueType = valuetype::Create();

  // Arrange
  Graph rvsdg;

  // Act & Assert
  // A newly created node without any users should automatically be added to the bottom nodes
  auto structuralNode = jlm::tests::structural_node::create(&rvsdg.GetRootRegion(), 1);
  assert(structuralNode->IsDead());
  assert(rvsdg.GetRootRegion().NumBottomNodes() == 1);
  assert(&*(rvsdg.GetRootRegion().BottomNodes().begin()) == structuralNode);

  // The node cedes to be dead
  auto & output = structuralNode->AddOutput(valueType);
  jlm::tests::GraphExport::Create(output, "x");
  assert(structuralNode->IsDead() == false);
  assert(rvsdg.GetRootRegion().NumBottomNodes() == 0);
  assert(rvsdg.GetRootRegion().BottomNodes().begin() == rvsdg.GetRootRegion().BottomNodes().end());

  // And it becomes dead again
  rvsdg.GetRootRegion().RemoveResultsWhere(
      [](const RegionResult &)
      {
        return true;
      });
  assert(structuralNode->IsDead());
  assert(rvsdg.GetRootRegion().NumBottomNodes() == 1);
  assert(&*(rvsdg.GetRootRegion().BottomNodes().begin()) == structuralNode);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-BottomNodeTests", BottomNodeTests)
