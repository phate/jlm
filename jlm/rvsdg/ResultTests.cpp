/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

#include <cassert>

/**
 * Test check for adding result to output of wrong structural node.
 */
TEST(ResultTests, ResultNodeMismatch)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto & import = jlm::rvsdg::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 1);

  auto input = structuralNode1->addInputWithArguments(import);

  // Act & Assert
  EXPECT_THROW(structuralNode2->addOutputWithResults({ input.argument[0] }), jlm::util::Error);
}

TEST(ResultTests, ResultInputTypeMismatch)
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto stateType = jlm::rvsdg::TestType::createStateType();

  jlm::rvsdg::Graph rvsdg;
  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 2);
  auto simpleNode0 = TestOperation::createNode(structuralNode->subregion(0), {}, { stateType });
  auto simpleNode1 = TestOperation::createNode(structuralNode->subregion(1), {}, { valueType });

  // Act & Assert
  EXPECT_THROW(
      structuralNode->addOutputWithResults({ simpleNode0->output(0), simpleNode1->output(0) }),
      TypeError);
}
