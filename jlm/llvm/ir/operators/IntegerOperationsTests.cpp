/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::llvm
{

namespace
{
struct FoldConstantsTestInput
{
  std::int64_t c1;
  std::uint64_t numBitsC1;

  std::int64_t c2;
  std::uint64_t numBitsC2;

  std::int64_t expected;
  std::int64_t numBitsExpected;
};
}

template<typename Operation>
static void
TestFoldConstants(const FoldConstantsTestInput & input)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  auto & c1 = IntegerConstantOperation::Create(
      graph.GetRootRegion(),
      BitValueRepresentation(input.numBitsC1, input.c1));
  auto & c2 = IntegerConstantOperation::Create(
      graph.GetRootRegion(),
      BitValueRepresentation(input.numBitsC2, input.c2));

  auto & node1 = Operation::createNode(32, *c1.output(0), *c2.output(0));

  auto & x1 = GraphExport::Create(*node1.output(0), "x1");

  view(graph, stdout);

  // Act
  ReduceNode<Operation>(Operation::foldConstants, dynamic_cast<SimpleNode &>(node1));

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_int(), input.expected);
    EXPECT_EQ(op->Representation().nbits(), static_cast<size_t>(input.numBitsExpected));
  }
}

TEST(IntegerEqOperationTest, foldConstants)
{
  TestFoldConstants<IntegerEqOperation>({ 4, 32, 4, 32, -1, 1 });
  TestFoldConstants<IntegerEqOperation>({ 4, 32, -4, 32, 0, 1 });
}

TEST(IntegerNeOperationTest, foldConstants)
{
  TestFoldConstants<IntegerNeOperation>({ 4, 32, 4, 32, 0, 1 });
  TestFoldConstants<IntegerNeOperation>({ 4, 32, -4, 32, -1, 1 });
}

TEST(IntegerSgeOperationTest, foldConstants)
{
  TestFoldConstants<IntegerSgeOperation>({ 4, 32, 4, 32, -1, 1 });
  TestFoldConstants<IntegerSgeOperation>({ 4, 32, -4, 32, -1, 1 });
}

TEST(IntegerSgtOperationTest, foldConstants)
{
  TestFoldConstants<IntegerSgtOperation>({ 4, 32, 4, 32, 0, 1 });
  TestFoldConstants<IntegerSgtOperation>({ 4, 32, -4, 32, -1, 1 });
}

TEST(IntegerSleOperationTest, foldConstants)
{
  TestFoldConstants<IntegerSleOperation>({ 4, 32, 4, 32, -1, 1 });
  TestFoldConstants<IntegerSleOperation>({ 4, 32, -4, 32, 0, 1 });
}

TEST(IntegerSltOperationTest, foldConstants)
{
  TestFoldConstants<IntegerSltOperation>({ 4, 32, 4, 32, 0, 1 });
  TestFoldConstants<IntegerSltOperation>({ 4, 32, -4, 32, 0, 1 });
}

TEST(IntegerUgeOperationTest, foldConstants)
{
  TestFoldConstants<IntegerUgeOperation>({ 4, 32, 4, 32, -1, 1 });
  TestFoldConstants<IntegerUgeOperation>({ 4, 32, -4, 32, 0, 1 });
}

TEST(IntegerUgtOperationTest, foldConstants)
{
  TestFoldConstants<IntegerUgtOperation>({ 4, 32, 4, 32, 0, 1 });
  TestFoldConstants<IntegerUgtOperation>({ 4, 32, -4, 32, 0, 1 });
}

TEST(IntegerUleOperationTest, foldConstants)
{
  TestFoldConstants<IntegerUleOperation>({ 4, 32, 4, 32, -1, 1 });
  TestFoldConstants<IntegerUleOperation>({ 4, 32, -4, 32, -1, 1 });
}

TEST(IntegerUltOperationTest, foldConstants)
{
  TestFoldConstants<IntegerUltOperation>({ 4, 32, 4, 32, 0, 1 });
  TestFoldConstants<IntegerUltOperation>({ 4, 32, -4, 32, -1, 1 });
}

TEST(IntegerOrOperationTest, foldConstants)
{
  TestFoldConstants<IntegerOrOperation>({ 4, 32, 4, 32, 4, 32 });
  TestFoldConstants<IntegerOrOperation>({ 0, 32, -1, 32, -1, 32 });
  TestFoldConstants<IntegerOrOperation>({ 1, 32, 2, 32, 3, 32 });
}

TEST(IntegerAndOperationTest, foldConstants)
{
  TestFoldConstants<IntegerAndOperation>({ 4, 32, 4, 32, 4, 32 });
  TestFoldConstants<IntegerAndOperation>({ 0, 32, -1, 32, 0, 32 });
}

TEST(IntegerXorOperationTest, foldConstants)
{
  TestFoldConstants<IntegerXorOperation>({ 4, 32, 4, 32, 0, 32 });
  TestFoldConstants<IntegerXorOperation>({ 0, 32, -1, 32, -1, 32 });
}

TEST(IntegerSubOperationTests, normalizeAdditiveInverse)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i32Type = BitType::Create(32);

  Graph graph;

  auto & i0 = GraphImport::Create(graph, i32Type, "i0");

  auto structuralNode = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto inputVar = structuralNode->addInputWithArguments(i0);

  auto & subNode =
      IntegerSubOperation::createNode(32, *inputVar.argument[0], *inputVar.argument[0]);

  auto outputVar = structuralNode->addOutputWithResults({ subNode.output(0) });

  GraphExport::Create(*outputVar.output, "x0");

  // Act
  ReduceNode<IntegerSubOperation>(
      IntegerSubOperation::normalizeAdditiveInverse,
      dynamic_cast<SimpleNode &>(subNode));

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] =
        TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*outputVar.result[0]->origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_int(), 0u);
    EXPECT_EQ(op->Representation().nbits(), 32u);
  }
}

}
