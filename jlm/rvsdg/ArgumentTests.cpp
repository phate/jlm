/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestType.hpp>

#include <cassert>

/**
 * Test check for adding a region argument to input of wrong structural node.
 */
TEST(ArgumentTests, ArgumentNodeMismatch)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto & import = jlm::rvsdg::GraphImport::Create(graph, valueType, "import");

  auto structuralNode1 = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto structuralNode2 = TestStructuralNode::create(&graph.GetRootRegion(), 2);

  auto & structuralInput = structuralNode1->addInputOnly(import);

  // Act & Assert
  EXPECT_THROW(
      RegionArgument::Create(*structuralNode2->subregion(0), &structuralInput, valueType),
      jlm::util::Error);
}

TEST(ArgumentTests, ArgumentInputTypeMismatch)
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = TestType::createValueType();
  auto stateType = TestType::createStateType();

  jlm::rvsdg::Graph rvsdg;
  auto & x = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "import");

  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 1);
  auto & structuralInput = structuralNode->addInputOnly(x);

  // Act & Assert
  EXPECT_THROW(
      RegionArgument::Create(*structuralNode->subregion(0), &structuralInput, stateType),
      jlm::util::TypeError);
}
