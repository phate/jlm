/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestType.hpp>

#include <cassert>

TEST(StructuralNodeTests, TestOutputRemoval)
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
  EXPECT_EQ(structuralNode->noutputs(), 5u);
  EXPECT_EQ(output0.index(), 0u);
  EXPECT_EQ(output1.index(), 1u);
  EXPECT_EQ(output2.index(), 2u);
  EXPECT_EQ(output3.index(), 3u);
  EXPECT_EQ(output4.index(), 4u);

  structuralNode->removeOutputAndResults(2);
  EXPECT_EQ(structuralNode->noutputs(), 4u);
  EXPECT_EQ(output0.index(), 0u);
  EXPECT_EQ(output1.index(), 1u);
  EXPECT_EQ(output3.index(), 2u);
  EXPECT_EQ(output4.index(), 3u);

  structuralNode->removeOutputAndResults(3);
  EXPECT_EQ(structuralNode->noutputs(), 3u);
  EXPECT_EQ(output0.index(), 0u);
  EXPECT_EQ(output1.index(), 1u);
  EXPECT_EQ(output3.index(), 2u);
}
