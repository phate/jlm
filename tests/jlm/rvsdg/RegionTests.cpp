/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <cassert>

/**
 * Test region::Contains().
 */
static int
Contains()
{
  using namespace jlm::tests;

  // Arrange
  auto valueType = valuetype::Create();

  jlm::rvsdg::graph graph;
  auto import = graph.add_import({ valueType, "import" });

  auto structuralNode1 = structural_node::create(graph.root(), 1);
  auto structuralInput1 = jlm::rvsdg::structural_input::create(structuralNode1, import, valueType);
  auto regionArgument1 =
      jlm::rvsdg::argument::create(structuralNode1->subregion(0), structuralInput1, valueType);
  unary_op::create(structuralNode1->subregion(0), valueType, regionArgument1, valueType);

  auto structuralNode2 = structural_node::create(graph.root(), 1);
  auto structuralInput2 = jlm::rvsdg::structural_input::create(structuralNode2, import, valueType);
  auto regionArgument2 =
      jlm::rvsdg::argument::create(structuralNode2->subregion(0), structuralInput2, valueType);
  binary_op::create(valueType, valueType, regionArgument2, regionArgument2);

  // Act & Assert
  assert(jlm::rvsdg::region::Contains<structural_op>(*graph.root(), false));
  assert(jlm::rvsdg::region::Contains<unary_op>(*graph.root(), true));
  assert(jlm::rvsdg::region::Contains<binary_op>(*graph.root(), true));
  assert(!jlm::rvsdg::region::Contains<test_op>(*graph.root(), true));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-Contains", Contains)

/**
 * Test region::IsRootRegion().
 */
static int
IsRootRegion()
{
  // Arrange
  jlm::rvsdg::graph graph;

  auto structuralNode = jlm::tests::structural_node::create(graph.root(), 1);

  // Act & Assert
  assert(graph.root()->IsRootRegion());
  assert(!structuralNode->subregion(0)->IsRootRegion());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-IsRootRegion", IsRootRegion)

/**
 * Test region::NumRegions() with an empty Rvsdg.
 */
static int
NumRegions_EmptyRvsdg()
{
  using namespace jlm::rvsdg;

  // Arrange
  jlm::rvsdg::graph graph;

  // Act & Assert
  assert(region::NumRegions(*graph.root()) == 1);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-NumRegions_EmptyRvsdg", NumRegions_EmptyRvsdg)

/**
 * Test region::NumRegions() with non-empty Rvsdg.
 */
static int
NumRegions_NonEmptyRvsdg()
{
  using namespace jlm::rvsdg;

  // Arrange
  jlm::rvsdg::graph graph;
  auto structuralNode = jlm::tests::structural_node::create(graph.root(), 4);
  jlm::tests::structural_node::create(structuralNode->subregion(0), 2);
  jlm::tests::structural_node::create(structuralNode->subregion(3), 5);

  // Act & Assert
  assert(region::NumRegions(*graph.root()) == 1 + 4 + 2 + 5);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-NumRegions_NonEmptyRvsdg", NumRegions_NonEmptyRvsdg)

/**
 * Test region::RemoveResultsWhere()
 */
static int
RemoveResultsWhere()
{
  // Arrange
  jlm::rvsdg::graph rvsdg;
  jlm::rvsdg::region region(rvsdg.root(), &rvsdg);

  auto valueType = jlm::tests::valuetype::Create();
  auto node = jlm::tests::test_op::Create(&region, {}, {}, { valueType });

  auto result0 =
      jlm::rvsdg::result::create(&region, node->output(0), nullptr, jlm::rvsdg::port(valueType));
  auto result1 =
      jlm::rvsdg::result::create(&region, node->output(0), nullptr, jlm::rvsdg::port(valueType));
  auto result2 =
      jlm::rvsdg::result::create(&region, node->output(0), nullptr, jlm::rvsdg::port(valueType));

  // Act & Arrange
  assert(region.nresults() == 3);
  assert(result0->index() == 0);
  assert(result1->index() == 1);
  assert(result2->index() == 2);

  region.RemoveResultsWhere(
      [](const jlm::rvsdg::result & result)
      {
        return result.index() == 1;
      });
  assert(region.nresults() == 2);
  assert(result0->index() == 0);
  assert(result2->index() == 1);

  region.RemoveResultsWhere(
      [](const jlm::rvsdg::result & result)
      {
        return false;
      });
  assert(region.nresults() == 2);
  assert(result0->index() == 0);
  assert(result2->index() == 1);

  region.RemoveResultsWhere(
      [](const jlm::rvsdg::result & result)
      {
        return true;
      });
  assert(region.nresults() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-RemoveResultsWhere", RemoveResultsWhere)

/**
 * Test region::RemoveArgumentsWhere()
 */
static int
RemoveArgumentsWhere()
{
  // Arrange
  jlm::rvsdg::graph rvsdg;
  jlm::rvsdg::region region(rvsdg.root(), &rvsdg);

  auto valueType = jlm::tests::valuetype::Create();
  auto argument0 = jlm::rvsdg::argument::create(&region, nullptr, jlm::rvsdg::port(valueType));
  auto argument1 = jlm::rvsdg::argument::create(&region, nullptr, jlm::rvsdg::port(valueType));
  auto argument2 = jlm::rvsdg::argument::create(&region, nullptr, jlm::rvsdg::port(valueType));

  auto node = jlm::tests::test_op::Create(&region, { valueType }, { argument1 }, { valueType });

  // Act & Arrange
  assert(region.narguments() == 3);
  assert(argument0->index() == 0);
  assert(argument1->index() == 1);
  assert(argument2->index() == 2);

  region.RemoveArgumentsWhere(
      [](const jlm::rvsdg::argument & argument)
      {
        return true;
      });
  assert(region.narguments() == 1);
  assert(argument1->index() == 0);

  region.remove_node(node);
  region.RemoveArgumentsWhere(
      [](const jlm::rvsdg::argument & argument)
      {
        return false;
      });
  assert(region.narguments() == 1);
  assert(argument1->index() == 0);

  region.RemoveArgumentsWhere(
      [](const jlm::rvsdg::argument & argument)
      {
        return argument.index() == 0;
      });
  assert(region.narguments() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-RemoveArgumentsWhere", RemoveArgumentsWhere)

/**
 * Test region::PruneArguments()
 */
static int
PruneArguments()
{
  // Arrange
  jlm::rvsdg::graph rvsdg;
  jlm::rvsdg::region region(rvsdg.root(), &rvsdg);

  auto valueType = jlm::tests::valuetype::Create();
  auto argument0 = jlm::rvsdg::argument::create(&region, nullptr, jlm::rvsdg::port(valueType));
  jlm::rvsdg::argument::create(&region, nullptr, jlm::rvsdg::port(valueType));
  auto argument2 = jlm::rvsdg::argument::create(&region, nullptr, jlm::rvsdg::port(valueType));

  auto node = jlm::tests::test_op::Create(
      &region,
      { valueType, valueType },
      { argument0, argument2 },
      { valueType });

  // Act & Arrange
  assert(region.narguments() == 3);

  region.PruneArguments();
  assert(region.narguments() == 2);
  assert(argument0->index() == 0);
  assert(argument2->index() == 1);

  region.remove_node(node);
  region.PruneArguments();
  assert(region.narguments() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-PruneArguments", PruneArguments)

static int
TestToTree_EmptyRvsdg()
{
  using namespace jlm::rvsdg;

  // Arrange
  graph rvsdg;

  // Act
  auto tree = region::ToTree(*rvsdg.root());
  std::cout << tree << std::flush;

  // Assert
  assert(tree == "RootRegion\n");

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/RegionTests-TestToTree_EmptyRvsdg", TestToTree_EmptyRvsdg)

static int
TestToTree_RvsdgWithStructuralNodes()
{
  using namespace jlm::rvsdg;

  // Arrange
  graph rvsdg;
  auto structuralNode = jlm::tests::structural_node::create(rvsdg.root(), 2);
  jlm::tests::structural_node::create(structuralNode->subregion(1), 3);

  // Act
  auto tree = region::ToTree(*rvsdg.root());
  std::cout << tree << std::flush;

  // Assert
  auto numLines = std::count(tree.begin(), tree.end(), '\n');

  // We should find '\n' 8 times: 1 root region + 2 structural nodes + 5 subregions
  assert(numLines == 8);

  // Check that the last line printed looks accordingly
  auto lastLine = std::string("----Region[2]\n");
  assert(tree.compare(tree.size() - lastLine.size(), lastLine.size(), lastLine) == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/RegionTests-TestToTree_RvsdgWithStructuralNodes",
    TestToTree_RvsdgWithStructuralNodes)
