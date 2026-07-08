/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/ControlOperations.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::llvm
{

TEST(ControlOperationsTests, foldConstants)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bitType32 = BitType::Create(32);

  auto & zeroNode =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(32, 0));
  auto & matchNode1 = MatchOperation::CreateNode(*zeroNode.output(0), { { 0, 0 } }, 1, 2);

  auto & fourNode =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(32, 4));
  auto & matchNode2 = MatchOperation::CreateNode(*fourNode.output(0), { { 0, 0 } }, 1, 2);

  auto & x1 = GraphExport::Create(*matchNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*matchNode2.output(0), "x2");

  view(graph, stdout);

  // Act
  ReduceNode<MatchOperation>(
      foldMatchOperationWithConstant,
      dynamic_cast<SimpleNode &>(matchNode1));

  ReduceNode<MatchOperation>(
      foldMatchOperationWithConstant,
      dynamic_cast<SimpleNode &>(matchNode2));

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<ControlConstantOperation>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->value().nalternatives(), 2u);
    EXPECT_EQ(op->value().alternative(), 0u);
  }

  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<ControlConstantOperation>(*x2.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->value().nalternatives(), 2u);
    EXPECT_EQ(op->value().alternative(), 1u);
  }
}

}
