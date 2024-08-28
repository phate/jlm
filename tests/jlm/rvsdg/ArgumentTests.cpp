/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <cassert>

/**
 * Test check for adding argument to input of wrong structural node.
 */
static int
ArgumentNodeMismatch()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto import = &jlm::tests::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = jlm::tests::structural_node::create(graph.root(), 1);
  auto structuralNode2 = jlm::tests::structural_node::create(graph.root(), 2);

  auto structuralInput = structural_input::create(structuralNode1, import, valueType);

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

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/ArgumentTests-ArgumentNodeMismatch", ArgumentNodeMismatch)

static int
ArgumentInputTypeMismatch()
{
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto stateType = jlm::tests::statetype::Create();

  jlm::rvsdg::graph rvsdg;
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "import");

  auto structuralNode = structural_node::create(rvsdg.root(), 1);
  auto structuralInput = jlm::rvsdg::structural_input::create(structuralNode, x, valueType);

  // Act & Assert
  bool exceptionWasCaught = false;
  try
  {
    TestGraphArgument::Create(*structuralNode->subregion(0), structuralInput, stateType);
    // The line below should not be executed as the line above is expected to throw an exception.
    assert(false);
  }
  catch (type_error &)
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
  catch (type_error &)
  {
    exceptionWasCaught = true;
  }
  assert(exceptionWasCaught);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/ArgumentTests-ArgumentInputTypeMismatch",
    ArgumentInputTypeMismatch)
