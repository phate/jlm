/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <cassert>

/**
 * Test check for adding argument to input of wrong structural node.
 */
static void
TestArgumentNodeMismatch()
{
  using namespace jlm::rvsdg;

  auto vt = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto import = graph.add_import({ vt, "import" });

  auto structuralNode1 = jlm::tests::structural_node::create(graph.root(), 1);
  auto structuralNode2 = jlm::tests::structural_node::create(graph.root(), 2);

  auto structuralInput = structural_input::create(structuralNode1, import, vt);

  bool inputErrorHandlerCalled = false;
  try
  {
    argument::create(structuralNode2->subregion(0), structuralInput, vt);
  }
  catch (jlm::util::error & e)
  {
    inputErrorHandlerCalled = true;
  }

  assert(inputErrorHandlerCalled);
}

/**
 * Test check for adding result to output of wrong structural node.
 */
static void
TestResultNodeMismatch()
{
  using namespace jlm::rvsdg;

  auto vt = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto import = graph.add_import({ vt, "import" });

  auto structuralNode1 = jlm::tests::structural_node::create(graph.root(), 1);
  auto structuralNode2 = jlm::tests::structural_node::create(graph.root(), 2);

  auto structuralInput = structural_input::create(structuralNode1, import, vt);

  auto argument = argument::create(structuralNode1->subregion(0), structuralInput, vt);
  auto structuralOutput = structural_output::create(structuralNode1, vt);

  bool outputErrorHandlerCalled = false;
  try
  {
    result::create(structuralNode2->subregion(0), argument, structuralOutput, vt);
  }
  catch (jlm::util::error & e)
  {
    outputErrorHandlerCalled = true;
  }

  assert(outputErrorHandlerCalled);
}

/**
 * Test region::Contains().
 */
static void
TestContainsMethod()
{
  using namespace jlm::tests;

  auto vt = valuetype::Create();

  jlm::rvsdg::graph graph;
  auto import = graph.add_import({ vt, "import" });

  auto structuralNode1 = structural_node::create(graph.root(), 1);
  auto structuralInput1 = jlm::rvsdg::structural_input::create(structuralNode1, import, vt);
  auto regionArgument1 =
      jlm::rvsdg::argument::create(structuralNode1->subregion(0), structuralInput1, vt);
  unary_op::create(structuralNode1->subregion(0), vt, regionArgument1, vt);

  auto structuralNode2 = structural_node::create(graph.root(), 1);
  auto structuralInput2 = jlm::rvsdg::structural_input::create(structuralNode2, import, vt);
  auto regionArgument2 =
      jlm::rvsdg::argument::create(structuralNode2->subregion(0), structuralInput2, vt);
  binary_op::create(vt, vt, regionArgument2, regionArgument2);

  assert(jlm::rvsdg::region::Contains<structural_op>(*graph.root(), false));
  assert(jlm::rvsdg::region::Contains<unary_op>(*graph.root(), true));
  assert(jlm::rvsdg::region::Contains<binary_op>(*graph.root(), true));
  assert(!jlm::rvsdg::region::Contains<test_op>(*graph.root(), true));
}

/**
 * Test region::IsRootRegion().
 */
static void
TestIsRootRegion()
{
  jlm::rvsdg::graph graph;

  auto structuralNode = jlm::tests::structural_node::create(graph.root(), 1);

  assert(graph.root()->IsRootRegion());
  assert(!structuralNode->subregion(0)->IsRootRegion());
}

/**
 * Test region::NumRegions()
 */
static void
TestNumRegions()
{
  using namespace jlm::rvsdg;

  {
    jlm::rvsdg::graph graph;

    assert(region::NumRegions(*graph.root()) == 1);
  }

  {
    jlm::rvsdg::graph graph;
    auto structuralNode = jlm::tests::structural_node::create(graph.root(), 4);
    jlm::tests::structural_node::create(structuralNode->subregion(0), 2);
    jlm::tests::structural_node::create(structuralNode->subregion(3), 5);

    assert(region::NumRegions(*graph.root()) == 1 + 4 + 2 + 5);
  }
}

/**
 * Test region::RemoveResultsWhere()
 */
static void
TestRemoveResultsWhere()
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
}

/**
 * Test region::RemoveArgumentsWhere()
 */
static void
TestRemoveArgumentsWhere()
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
}

/**
 * Test region::PruneArguments()
 */
static void
TestPruneArguments()
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
}

static int
Test()
{
  TestArgumentNodeMismatch();
  TestResultNodeMismatch();

  TestContainsMethod();
  TestIsRootRegion();
  TestNumRegions();
  TestRemoveResultsWhere();
  TestRemoveArgumentsWhere();
  TestPruneArguments();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TestRegion", Test)

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

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TestRegion-TestToTree_EmptyRvsdg", TestToTree_EmptyRvsdg)

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
    "jlm/rvsdg/TestRegion-TestToTree_RvsdgWithStructuralNodes",
    TestToTree_RvsdgWithStructuralNodes)
