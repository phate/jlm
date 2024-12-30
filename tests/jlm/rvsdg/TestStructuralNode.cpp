/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <cassert>

static void
TestOutputRemoval()
{
  using namespace jlm;

  // Arrange
  rvsdg::Graph rvsdg;
  auto valueType = tests::valuetype::Create();

  auto structuralNode = tests::structural_node::create(&rvsdg.GetRootRegion(), 1);
  auto output0 = rvsdg::StructuralOutput::create(structuralNode, valueType);
  auto output1 = rvsdg::StructuralOutput::create(structuralNode, valueType);
  auto output2 = rvsdg::StructuralOutput::create(structuralNode, valueType);
  auto output3 = rvsdg::StructuralOutput::create(structuralNode, valueType);
  auto output4 = rvsdg::StructuralOutput::create(structuralNode, valueType);

  // Act & Assert
  assert(structuralNode->noutputs() == 5);
  assert(output0->index() == 0);
  assert(output1->index() == 1);
  assert(output2->index() == 2);
  assert(output3->index() == 3);
  assert(output4->index() == 4);

  structuralNode->RemoveOutput(2);
  assert(structuralNode->noutputs() == 4);
  assert(output0->index() == 0);
  assert(output1->index() == 1);
  assert(output3->index() == 2);
  assert(output4->index() == 3);

  structuralNode->RemoveOutput(3);
  assert(structuralNode->noutputs() == 3);
  assert(output0->index() == 0);
  assert(output1->index() == 1);
  assert(output3->index() == 2);
}

static int
TestStructuralNode()
{
  TestOutputRemoval();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TestStructuralNode", TestStructuralNode)
