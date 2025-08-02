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
  auto valueType = jlm::tests::ValueType::Create();

  Graph graph;
  auto import = &jlm::tests::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 1);

  auto structuralInput = StructuralInput::create(structuralNode1, import, valueType);

  auto & argument =
      TestGraphArgument::Create(*structuralNode1->subregion(0), structuralInput, valueType);

  // Act
  bool outputErrorHandlerCalled = false;
  try
  {
    // Region mismatch
    structuralNode2->AddOutputWithResults({ &argument });
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
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = ValueType::Create();
  auto stateType = StateType::Create();

  jlm::rvsdg::Graph rvsdg;
  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 2);

  // Act & Assert
  bool exceptionWasCaught = false;
  try
  {
    auto simpleNode0 = TestOperation::create(structuralNode->subregion(0), {}, { stateType });
    auto simpleNode1 = TestOperation::create(structuralNode->subregion(1), {}, { valueType });

    // Type mismatch between simple node output and structural output
    structuralNode->AddOutputWithResults({ simpleNode0->output(0), simpleNode1->output(0) });
    // The line below should not be executed as the line above is expected to throw an exception.
    assert(false);
  }
  catch (TypeError & error)
  {
    assert(std::string(error.what()) == "Type error - expected : ValueType, received : StateType");
    exceptionWasCaught = true;
  }
  assert(exceptionWasCaught);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/ResultTests-ResultInputTypeMismatch", ResultInputTypeMismatch)
