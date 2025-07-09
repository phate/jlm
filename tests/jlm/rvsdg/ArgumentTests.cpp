/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

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
  auto valueType = jlm::tests::ValueType::Create();

  Graph graph;
  auto import = &jlm::tests::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 2);

  auto structuralInput = StructuralInput::create(structuralNode1, import, valueType);

  // Act
  bool inputErrorHandlerCalled = false;
  try
  {
    TestGraphArgument::Create(*structuralNode2->subregion(0), structuralInput, valueType);
  }
  catch (jlm::util::error & e)
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
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = ValueType::Create();
  auto stateType = StateType::Create();

  jlm::rvsdg::Graph rvsdg;
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "import");

  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 1);
  auto structuralInput = jlm::rvsdg::StructuralInput::create(structuralNode, x, valueType);

  // Act & Assert
  bool exceptionWasCaught = false;
  try
  {
    TestGraphArgument::Create(*structuralNode->subregion(0), structuralInput, stateType);
    // The line below should not be executed as the line above is expected to throw an exception.
    assert(false);
  }
  catch (TypeError &)
  {
    exceptionWasCaught = true;
  }
  assert(exceptionWasCaught);

  exceptionWasCaught = false;
  try
  {
    TestGraphArgument::Create(*structuralNode->subregion(0), structuralInput, stateType);
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
