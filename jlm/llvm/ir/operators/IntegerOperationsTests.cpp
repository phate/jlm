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

TEST(IntegerEqOperationTest, foldConstants)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bitType32 = BitType::Create(32);

  auto & four =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(32, 4));
  auto & mfour =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(32, -4));

  auto & eqNode1 = IntegerEqOperation::createNode(32, *four.output(0), *four.output(0));
  auto & eqNode2 = IntegerEqOperation::createNode(32, *four.output(0), *mfour.output(0));

  auto & x1 = GraphExport::Create(*eqNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*eqNode2.output(0), "x1");

  view(graph, stdout);

  // Act
  ReduceNode<IntegerEqOperation>(
      IntegerEqOperation::foldConstants,
      dynamic_cast<SimpleNode &>(eqNode1));

  ReduceNode<IntegerEqOperation>(
      IntegerEqOperation::foldConstants,
      dynamic_cast<SimpleNode &>(eqNode2));

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_uint(), 1u);
    EXPECT_EQ(op->Representation().nbits(), 1u);
  }

  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x2.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_uint(), 0u);
    EXPECT_EQ(op->Representation().nbits(), 1u);
  }
}

}
