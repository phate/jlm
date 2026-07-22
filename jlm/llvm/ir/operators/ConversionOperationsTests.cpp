/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::llvm
{
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

TEST(ConversionOperationsTests, TruncConstantFolding)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto i1Type = BitType::Create(1);

  Graph graph;

  auto & eight =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(32, 8));
  auto & one =
      IntegerConstantOperation::Create(graph.GetRootRegion(), BitValueRepresentation(32, 1));

  auto & node1 = TruncOperation::createNode(*eight.output(0), i1Type);
  auto & node2 = TruncOperation::createNode(*one.output(0), i1Type);

  auto & x1 = GraphExport::Create(*node1.output(0), "x1");
  auto & x2 = GraphExport::Create(*node2.output(0), "x2");

  view(graph, stdout);

  // Act
  ReduceNode<TruncOperation>(TruncOperation::foldConstant, node1);
  ReduceNode<TruncOperation>(TruncOperation::foldConstant, node2);

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_uint(), 0u);
    EXPECT_EQ(op->Representation().nbits(), 1u);
  }

  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*x2.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(op->Representation().to_uint(), 1);
    EXPECT_EQ(op->Representation().nbits(), 1u);
  }
}

TEST(ConversionOperationsTests, FunctionToPointerInversion)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();
  auto pointerType = PointerType::Create();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  Graph graph;

  auto & i0 = GraphImport::Create(graph, pointerType, "i0");
  auto & i1 = GraphImport::Create(graph, functionType, "i1");

  auto & ptrToFnNode = rvsdg::CreateOpNode<PointerToFunctionOperation>({ &i0 }, functionType);

  auto & fnToPtrNode1 =
      rvsdg::CreateOpNode<FunctionToPointerOperation>({ ptrToFnNode.output(0) }, functionType);

  auto & fnToPtrNode2 = rvsdg::CreateOpNode<FunctionToPointerOperation>({ &i1 }, functionType);

  auto & x1 = GraphExport::Create(*fnToPtrNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*fnToPtrNode2.output(0), "x2");

  view(graph, stdout);

  // Act
  ReduceNode<FunctionToPointerOperation>(
      FunctionToPointerOperation::invertFunctionToPointer,
      fnToPtrNode1);

  ReduceNode<FunctionToPointerOperation>(
      FunctionToPointerOperation::invertFunctionToPointer,
      fnToPtrNode2);

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  // The transformation should have been successful.
  {
    EXPECT_EQ(x1.origin(), &i0);
  }

  // The transformation should have failed.
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<FunctionToPointerOperation>(*x2.origin());
    EXPECT_NE(op, nullptr);
  }
}

TEST(ConversionOperationsTests, PointerToFunctionInversion)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto stateType = TestType::createStateType();
  auto valueType = TestType::createValueType();
  auto pointerType = PointerType::Create();
  auto functionType1 = FunctionType::Create({ valueType }, { valueType });
  auto functionType2 = FunctionType::Create({ stateType }, { stateType });

  Graph graph;

  auto & i0 = GraphImport::Create(graph, functionType1, "i0");

  auto & fnToPtrNode = rvsdg::CreateOpNode<FunctionToPointerOperation>({ &i0 }, functionType1);

  auto & ptrToFnNode1 =
      rvsdg::CreateOpNode<PointerToFunctionOperation>({ fnToPtrNode.output(0) }, functionType1);

  auto & ptrToFnNode2 =
      rvsdg::CreateOpNode<PointerToFunctionOperation>({ fnToPtrNode.output(0) }, functionType2);

  auto & x1 = GraphExport::Create(*ptrToFnNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*ptrToFnNode2.output(0), "x2");

  view(graph, stdout);

  // Act
  ReduceNode<PointerToFunctionOperation>(
      PointerToFunctionOperation::invertPointerToFunction,
      ptrToFnNode1);

  ReduceNode<PointerToFunctionOperation>(
      PointerToFunctionOperation::invertPointerToFunction,
      ptrToFnNode2);

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  // The transformation should have been successful.
  {
    EXPECT_EQ(x1.origin(), &i0);
  }

  // The transformation should have failed as the function types of the FunctionToPointer and
  // PointerToFunction operations are different.
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<PointerToFunctionOperation>(*x2.origin());
    EXPECT_NE(op, nullptr);
  }
}

}
