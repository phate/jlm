/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/opt/NodeReduction.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/SimpleNode.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

TEST(ConversionOperationsTests, BitIntegerTypeConversions)
{
  using namespace jlm::rvsdg;

  Graph graph;
  const auto bitType = BitType::Create(8);
  const auto integerType = IntegerType::create(8);
  auto & constant =
      IntegerConstantOperation::Create(graph.GetRootRegion(), IntegerValueRepresentation(8, 42));

  auto & integerNode = BitToIntegerTypeOperation::createNode(*constant.output(0), integerType);
  auto & bitNode = IntegerToBitTypeOperation::createNode(*integerNode.output(0), bitType);

  EXPECT_EQ(*integerNode.output(0)->Type(), *integerType);
  EXPECT_EQ(*bitNode.output(0)->Type(), *bitType);

  const IntegerToBitTypeOperation inferredBitType(integerType);
  EXPECT_EQ(*inferredBitType.getIntegerType(), *integerType);
  EXPECT_EQ(*inferredBitType.getBitType(), *bitType);
  EXPECT_THROW((IntegerToBitTypeOperation(integerType, BitType::Create(16))), std::runtime_error);
}

TEST(ConversionOperationsTests, NormalizeInverseBitIntegerConversions)
{
  using namespace jlm::rvsdg;

  const auto bitType = BitType::Create(8);
  const auto integerType = IntegerType::create(8);

  LlvmRvsdgModule rvsdgModule(util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto & bitArgument = GraphImport::Create(graph, bitType, "bits");
  auto & integerArgument = GraphImport::Create(graph, integerType, "integer");

  auto & integerNode = BitToIntegerTypeOperation::createNode(bitArgument, integerType);
  auto & bitResultNode =
      IntegerToBitTypeOperation::createNode(*integerNode.output(0), bitType);
  auto & bitNode = IntegerToBitTypeOperation::createNode(integerArgument, bitType);
  auto & integerResultNode =
      BitToIntegerTypeOperation::createNode(*bitNode.output(0), integerType);

  GraphExport::Create(*bitResultNode.output(0), "bitsResult");
  GraphExport::Create(*integerResultNode.output(0), "integerResult");

  NodeReduction nodeReduction;
  util::StatisticsCollector statisticsCollector(
      util::StatisticsCollectorSettings({ util::Statistics::Id::ReduceNodes }));
  nodeReduction.Run(rvsdgModule, statisticsCollector);

  EXPECT_EQ(graph.GetRootRegion().result(0)->origin(), &bitArgument);
  EXPECT_EQ(graph.GetRootRegion().result(1)->origin(), &integerArgument);
}

TEST(ConversionOperationsTests, SExtConstantFolding)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  auto & zero =
      IntegerConstantOperation::Create(graph.GetRootRegion(), IntegerValueRepresentation(1, 0));
  auto & one =
      IntegerConstantOperation::Create(graph.GetRootRegion(), IntegerValueRepresentation(1, 1));

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
      IntegerConstantOperation::Create(graph.GetRootRegion(), IntegerValueRepresentation(1, 0));
  auto & one =
      IntegerConstantOperation::Create(graph.GetRootRegion(), IntegerValueRepresentation(1, 1));

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
      IntegerConstantOperation::Create(graph.GetRootRegion(), IntegerValueRepresentation(32, 8));
  auto & one =
      IntegerConstantOperation::Create(graph.GetRootRegion(), IntegerValueRepresentation(32, 1));

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
    EXPECT_EQ(op->Representation().to_uint(), 1u);
    EXPECT_EQ(op->Representation().nbits(), 1u);
  }
}

TEST(ConversionOperationsTests, fpExtConstantFolding)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto fpFltType = FloatingPointType::Create(fpsize::flt);
  auto fpDblType = FloatingPointType::Create(fpsize::dbl);

  Graph graph;

  auto & eight = ConstantFP::createNode(graph.GetRootRegion(), fpsize::flt, ::llvm::APFloat(8.0));
  auto & one = ConstantFP::createNode(graph.GetRootRegion(), fpsize::half, ::llvm::APFloat(1.0));

  auto & node1 = FPExtOperation::createNode(*eight.output(0), fpDblType);
  auto & node2 = FPExtOperation::createNode(*one.output(0), fpFltType);

  auto & x1 = GraphExport::Create(*node1.output(0), "x1");
  auto & x2 = GraphExport::Create(*node2.output(0), "x2");

  view(graph, stdout);

  // Act
  ReduceNode<FPExtOperation>(FPExtOperation::foldConstant, node1);
  ReduceNode<FPExtOperation>(FPExtOperation::foldConstant, node2);

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<ConstantFP>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(&op->constant().getSemantics(), &::llvm::APFloat::IEEEdouble());
    EXPECT_EQ(op->constant().convertToDouble(), 8.0);
  }

  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<ConstantFP>(*x2.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(&op->constant().getSemantics(), &::llvm::APFloat::IEEEsingle());
    EXPECT_EQ(op->constant().convertToFloat(), 1.0);
  }
}

TEST(ConversionOperationsTests, fpTruncConstantFolding)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto fpFltType = FloatingPointType::Create(fpsize::flt);

  Graph graph;

  auto & eight = ConstantFP::createNode(graph.GetRootRegion(), fpsize::dbl, ::llvm::APFloat(8.0));

  auto & node1 = FPTruncOperation::createNode(*eight.output(0), fpFltType);

  auto & x1 = GraphExport::Create(*node1.output(0), "x1");

  view(graph, stdout);

  // Act
  ReduceNode<FPTruncOperation>(FPTruncOperation::foldConstant, node1);

  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  {
    auto [_, op] = TryGetSimpleNodeAndOptionalOp<ConstantFP>(*x1.origin());
    EXPECT_TRUE(op);
    EXPECT_EQ(&op->constant().getSemantics(), &::llvm::APFloat::IEEEsingle());
    EXPECT_EQ(op->constant().convertToDouble(), 8.0);
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
