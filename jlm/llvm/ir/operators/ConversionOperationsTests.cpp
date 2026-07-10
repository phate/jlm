/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::llvm
{

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
  auto & zext = ZExtOperation::create(32, x);
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
    EXPECT_EQ(op->value().nbits(), 32u);
  }

  {
    // Check zext
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(*zextExport.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->value().to_int(), 130u);
    EXPECT_EQ(op->value().nbits(), 32u);
  }
}

TEST(ConversionOperationsTests, SExtConstantFolding)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  auto & zero =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(1, 0));
  auto & one =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(1, 1));

  auto & node1 = SExtOperation::createNode(32, *zero.output(0));
  auto & node2 = SExtOperation::createNode(32, *one.output(0));

  auto & x1 = GraphExport::Create(*node1.output(0), "x1");
  auto & x2 = GraphExport::Create(*node2.output(0), "x2");

  view(graph, stdout);

  // Act
  ReduceNode<SExtOperation>(SExtOperation::foldConstant, node1);
  ReduceNode<SExtOperation>(SExtOperation::foldConstant, node2);

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_uint(), 0u);
    EXPECT_EQ(op->Representation().nbits(), 32u);
  }

  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x2.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_int(), -1);
    EXPECT_EQ(op->Representation().nbits(), 32u);
  }
}

}

TEST(ConversionOperationsTests, ZExtConstantFolding)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  auto & zero =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(1, 0));
  auto & one =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(1, 1));

  auto & node1 = ZExtOperation::createNode(32, *zero.output(0));
  auto & node2 = ZExtOperation::createNode(32, *one.output(0));

  auto & x1 = GraphExport::Create(*node1.output(0), "x1");
  auto & x2 = GraphExport::Create(*node2.output(0), "x2");

  view(graph, stdout);

  // Act
  ReduceNode<ZExtOperation>(ZExtOperation::foldConstant, node1);
  ReduceNode<ZExtOperation>(ZExtOperation::foldConstant, node2);

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_uint(), 0u);
    EXPECT_EQ(op->Representation().nbits(), 32u);
  }

  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x2.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_int(), 1);
    EXPECT_EQ(op->Representation().nbits(), 32u);
  }
}

}
