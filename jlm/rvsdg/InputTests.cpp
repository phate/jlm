/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(InputTests, TestInputIterator)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  auto i = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<TestOperation>(
      { i, i, i, i, i },
      std::vector<std::shared_ptr<const Type>>(5, valueType),
      std::vector<std::shared_ptr<const Type>>{ valueType });

  GraphExport::Create(*node.output(0), "x0");
  GraphExport::Create(*node.output(0), "x1");
  GraphExport::Create(*node.output(0), "x2");

  // Act & Assert
  auto nodeIt = Input::Iterator(node.input(0));
  EXPECT_EQ(nodeIt.GetInput(), node.input(0));
  EXPECT_EQ(nodeIt->index(), node.input(0)->index());
  EXPECT_EQ((*nodeIt).index(), node.input(0)->index());
  EXPECT_EQ(nodeIt, Input::Iterator(node.input(0)));
  EXPECT_NE(nodeIt, Input::Iterator(node.input(1)));

  nodeIt++;
  EXPECT_EQ(nodeIt.GetInput(), node.input(1));

  ++nodeIt;
  EXPECT_EQ(nodeIt.GetInput(), node.input(2));

  ++nodeIt;
  ++nodeIt;
  EXPECT_EQ(nodeIt.GetInput(), node.input(4));

  ++nodeIt;
  EXPECT_EQ(nodeIt.GetInput(), nullptr);

  auto regionIt = Input::Iterator(rootRegion.result(0));
  EXPECT_EQ(regionIt.GetInput(), rootRegion.result(0));
  EXPECT_EQ(regionIt->index(), rootRegion.result(0)->index());
  EXPECT_EQ((*regionIt).index(), rootRegion.result(0)->index());
  EXPECT_EQ(regionIt, Input::Iterator(rootRegion.result(0)));
  EXPECT_NE(regionIt, Input::Iterator(rootRegion.result(1)));

  regionIt++;
  regionIt++;
  EXPECT_EQ(regionIt.GetInput(), rootRegion.result(2));

  regionIt++;
  EXPECT_EQ(regionIt.GetInput(), nullptr);

  auto it = Input::Iterator(nullptr);
  it++;
  ++it;
  EXPECT_EQ(it.GetInput(), nullptr);
}

TEST(InputTests, TestInputConstIterator)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  auto i = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<TestOperation>(
      { i, i, i, i, i },
      std::vector<std::shared_ptr<const Type>>(5, valueType),
      std::vector<std::shared_ptr<const Type>>{ valueType });

  GraphExport::Create(*node.output(0), "x0");
  GraphExport::Create(*node.output(0), "x1");
  GraphExport::Create(*node.output(0), "x2");

  // Act & Assert
  auto nodeIt = Input::ConstIterator(node.input(0));
  EXPECT_EQ(nodeIt.GetInput(), node.input(0));
  EXPECT_EQ(nodeIt->index(), node.input(0)->index());
  EXPECT_EQ((*nodeIt).index(), node.input(0)->index());
  EXPECT_EQ(nodeIt, Input::ConstIterator(node.input(0)));
  EXPECT_NE(nodeIt, Input::ConstIterator(node.input(1)));

  nodeIt++;
  EXPECT_EQ(nodeIt.GetInput(), node.input(1));

  ++nodeIt;
  EXPECT_EQ(nodeIt.GetInput(), node.input(2));

  ++nodeIt;
  ++nodeIt;
  EXPECT_EQ(nodeIt.GetInput(), node.input(4));

  ++nodeIt;
  EXPECT_EQ(nodeIt.GetInput(), nullptr);

  auto regionIt = Input::ConstIterator(rootRegion.result(0));
  EXPECT_EQ(regionIt.GetInput(), rootRegion.result(0));
  EXPECT_EQ(regionIt->index(), rootRegion.result(0)->index());
  EXPECT_EQ((*regionIt).index(), rootRegion.result(0)->index());
  EXPECT_EQ(regionIt, Input::ConstIterator(rootRegion.result(0)));
  EXPECT_NE(regionIt, Input::ConstIterator(rootRegion.result(1)));

  regionIt++;
  regionIt++;
  EXPECT_EQ(regionIt.GetInput(), rootRegion.result(2));

  regionIt++;
  EXPECT_EQ(regionIt.GetInput(), nullptr);

  auto it = Input::ConstIterator(nullptr);
  it++;
  ++it;
  EXPECT_EQ(it.GetInput(), nullptr);
}
