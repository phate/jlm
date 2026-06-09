/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(ConversionOperationsTests, testSextZextConstantReductions)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bitType32 = BitType::Create(32);

  // the 8-bit value 130 is -126 when interpreted as a signed value
  auto & x = BitConstantOperation::create(graph.GetRootRegion(), BitValueRepresentation(8, 130));
  auto & sext = jlm::llvm::SExtOperation::create(32, x);
  auto & zext = jlm::llvm::ZExtOperation::create(32, x);
  auto & sextExport = GraphExport::Create(sext, "sext");
  auto & zextExport = GraphExport::Create(zext, "zext");

  view(graph, stdout);

  // Act
  ReduceNode<SExtOperation>(
      NormalizeUnaryOperation,
      *jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(sext));
  graph.PruneNodes();
  ReduceNode<ZExtOperation>(
      NormalizeUnaryOperation,
      *jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(zext));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    // Check sext
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(*sextExport.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->value().to_int(), -126);
    EXPECT_EQ(op->value().nbits(), 32);
  }

  {
    // Check zext
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(*zextExport.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->value().to_int(), 130);
    EXPECT_EQ(op->value().nbits(), 32);
  }
}
