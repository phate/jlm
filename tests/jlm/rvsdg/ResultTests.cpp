/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

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
  auto valueType = TestType::createValueType();

  Graph graph;
  auto & import = jlm::rvsdg::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 1);

  auto input = structuralNode1->addInputWithArguments(import);

  // Act
  bool outputErrorHandlerCalled = false;
  try
  {
    // Region mismatch
    structuralNode2->addOutputWithResults({ input.argument[0] });
    // The line below should not be executed as the line above is expected to throw an exception.
    assert(false);
  }
  catch (jlm::util::Error & error)
  {
    assert(std::string(error.what()) == "Invalid operand region.");
    outputErrorHandlerCalled = true;
  }

  // Assert
  assert(outputErrorHandlerCalled);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/ResultTests-ResultNodeMismatch", ResultNodeMismatch)

static void
ResultInputTypeMismatch()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto stateType = jlm::rvsdg::TestType::createStateType();

  jlm::rvsdg::Graph rvsdg;
  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 2);

  // Act & Assert
  bool exceptionWasCaught = false;
  try
  {
    auto simpleNode0 = TestOperation::createNode(structuralNode->subregion(0), {}, { stateType });
    auto simpleNode1 = TestOperation::createNode(structuralNode->subregion(1), {}, { valueType });

    // Type mismatch between simple node output and structural output
    structuralNode->addOutputWithResults({ simpleNode0->output(0), simpleNode1->output(0) });
    // The line below should not be executed as the line above is expected to throw an exception.
    assert(false);
  }
  catch (TypeError & error)
  {
    assert(
        std::string(error.what())
        == "Type error - expected : TestType[Value], received : TestType[State]");
    exceptionWasCaught = true;
  }
  assert(exceptionWasCaught);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/ResultTests-ResultInputTypeMismatch", ResultInputTypeMismatch)
