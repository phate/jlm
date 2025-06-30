/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <cassert>

/**
 * Test check for adding result to output of wrong structural node.
 */
static void
ResultNodeMismatch()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  Graph graph;
  auto import = &jlm::tests::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = jlm::tests::structural_node::create(&graph.GetRootRegion(), 1);
  auto structuralNode2 = jlm::tests::structural_node::create(&graph.GetRootRegion(), 2);

  auto structuralInput = StructuralInput::create(structuralNode1, import, valueType);

  auto & argument =
      TestGraphArgument::Create(*structuralNode1->subregion(0), structuralInput, valueType);
  auto structuralOutput = StructuralOutput::create(structuralNode1, valueType);

  // Act
  bool outputErrorHandlerCalled = false;
  try
  {
    // Region mismatch
    TestGraphResult::Create(*structuralNode2->subregion(0), argument, structuralOutput);
  }
  catch (jlm::util::error & e)
  {
    outputErrorHandlerCalled = true;
  }

  // Assert
  assert(outputErrorHandlerCalled);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/ResultTests-ResultNodeMismatch", ResultNodeMismatch)

static void
ResultInputTypeMismatch()
{
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto stateType = StateType::Create();

  jlm::rvsdg::Graph rvsdg;

  auto structuralNode = structural_node::create(&rvsdg.GetRootRegion(), 1);
  auto structuralOutput = jlm::rvsdg::StructuralOutput::create(structuralNode, valueType);

  // Act & Assert
  bool exceptionWasCaught = false;
  try
  {
    auto simpleNode = test_op::create(structuralNode->subregion(0), {}, { stateType });

    // Type mismatch between simple node output and structural output
    TestGraphResult::Create(*simpleNode->output(0), structuralOutput);
    // The line below should not be executed as the line above is expected to throw an exception.
    assert(false);
  }
  catch (TypeError &)
  {
    exceptionWasCaught = true;
  }
  assert(exceptionWasCaught);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/ResultTests-ResultInputTypeMismatch", ResultInputTypeMismatch)
