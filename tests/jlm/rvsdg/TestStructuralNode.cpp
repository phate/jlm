/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestType.hpp>

#include <cassert>

static void
TestOutputRemoval()
{
  using namespace jlm;

  // Arrange
  rvsdg::Graph rvsdg;
  auto valueType = rvsdg::TestType::createValueType();

  auto structuralNode = rvsdg::TestStructuralNode::create(&rvsdg.GetRootRegion(), 1);
  auto & output0 = structuralNode->addOutputOnly(valueType);
  auto & output1 = structuralNode->addOutputOnly(valueType);
  auto & output2 = structuralNode->addOutputOnly(valueType);
  auto & output3 = structuralNode->addOutputOnly(valueType);
  auto & output4 = structuralNode->addOutputOnly(valueType);

  // Act & Assert
  assert(structuralNode->noutputs() == 5);
  assert(output0.index() == 0);
  assert(output1.index() == 1);
  assert(output2.index() == 2);
  assert(output3.index() == 3);
  assert(output4.index() == 4);

  structuralNode->removeOutputAndResults(2);
  assert(structuralNode->noutputs() == 4);
  assert(output0.index() == 0);
  assert(output1.index() == 1);
  assert(output3.index() == 2);
  assert(output4.index() == 3);

  structuralNode->removeOutputAndResults(3);
  assert(structuralNode->noutputs() == 3);
  assert(output0.index() == 0);
  assert(output1.index() == 1);
  assert(output3.index() == 2);
}

static void
TestStructuralNode()
{
  TestOutputRemoval();
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TestStructuralNode", TestStructuralNode)
