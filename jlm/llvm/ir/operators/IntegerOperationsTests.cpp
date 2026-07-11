/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::llvm
{

template<typename Operation>
static void
TestFoldConstants(
    const std::uint64_t expectedSameOperands,
    const std::uint64_t expectedDifferentOperands)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  auto & four =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(32, 4));
  auto & mfour =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(32, -4));

  auto & node1 = Operation::createNode(32, *four.output(0), *four.output(0));
  auto & node2 = Operation::createNode(32, *four.output(0), *mfour.output(0));

  auto & x1 = GraphExport::Create(*node1.output(0), "x1");
  auto & x2 = GraphExport::Create(*node2.output(0), "x2");

  view(graph, stdout);

  // Act
  ReduceNode<Operation>(Operation::foldConstants, dynamic_cast<SimpleNode &>(node1));
  ReduceNode<Operation>(Operation::foldConstants, dynamic_cast<SimpleNode &>(node2));

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_uint(), expectedSameOperands);
    EXPECT_EQ(op->Representation().nbits(), 1u);
  }

  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x2.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_uint(), expectedDifferentOperands);
    EXPECT_EQ(op->Representation().nbits(), 1u);
  }
}

TEST(IntegerEqOperationTest, foldConstants)
{
  TestFoldConstants<IntegerEqOperation>(1u, 0u);
}

TEST(IntegerNeOperationTest, foldConstants)
{
  TestFoldConstants<IntegerNeOperation>(0u, 1u);
}

TEST(IntegerSgeOperationTest, foldConstants)
{
  TestFoldConstants<IntegerSgeOperation>(1u, 1u);
}

TEST(IntegerSgtOperationTest, foldConstants)
{
  TestFoldConstants<IntegerSgtOperation>(0u, 1u);
}

TEST(IntegerSleOperationTest, foldConstants)
{
  TestFoldConstants<IntegerSleOperation>(1u, 0u);
}

TEST(IntegerSltOperationTest, foldConstants)
{
  TestFoldConstants<IntegerSltOperation>(0u, 0u);
}

TEST(IntegerUgeOperationTest, foldConstants)
{
  TestFoldConstants<IntegerUgeOperation>(1u, 0u);
}

TEST(IntegerUgtOperationTest, foldConstants)
{
  TestFoldConstants<IntegerUgtOperation>(0u, 0u);
}

TEST(IntegerUleOperationTest, foldConstants)
{
  TestFoldConstants<IntegerUleOperation>(1u, 1u);
}

TEST(IntegerUltOperationTest, foldConstants)
{
  TestFoldConstants<IntegerUltOperation>(0u, 1u);
}

}
