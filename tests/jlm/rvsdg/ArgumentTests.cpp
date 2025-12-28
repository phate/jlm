/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestType.hpp>

#include <cassert>

/**
 * Test check for adding a region argument to input of wrong structural node.
 */
static void
ArgumentNodeMismatch()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto & import = jlm::rvsdg::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 2);

  auto & structuralInput = structuralNode1->addInputOnly(import);

  // Act
  bool inputErrorHandlerCalled = false;
  try
  {
    RegionArgument::Create(*structuralNode2->subregion(0), &structuralInput, valueType);
  }
  catch (jlm::util::Error &)
  {
    inputErrorHandlerCalled = true;
  }

  // Assert
  assert(inputErrorHandlerCalled);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/ArgumentTests-ArgumentNodeMismatch", ArgumentNodeMismatch)

static void
ArgumentInputTypeMismatch()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = TestType::createValueType();
  auto stateType = TestType::createStateType();

  jlm::rvsdg::Graph rvsdg;
  auto & x = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "import");

  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 1);
  auto & structuralInput = structuralNode->addInputOnly(x);

  // Act & Assert
  bool exceptionWasCaught = false;
  try
  {
    RegionArgument::Create(*structuralNode->subregion(0), &structuralInput, stateType);
    // The line below should not be executed as the line above is expected to throw an exception.
    assert(false);
  }
  catch (TypeError &)
  {
    exceptionWasCaught = true;
  }
  assert(exceptionWasCaught);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/ArgumentTests-ArgumentInputTypeMismatch",
    ArgumentInputTypeMismatch)
