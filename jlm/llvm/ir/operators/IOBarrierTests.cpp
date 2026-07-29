/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>

namespace jlm::llvm
{

TEST(IOBarrierTests, normalizeIOBarrierFromKnownAddress)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto pointerType = PointerType::Create();
  auto i32Type = BitType::Create(32);
  auto ioStateType = IOStateType::Create();
  auto functionType = FunctionType::Create({}, { pointerType });

  Graph graph;

  auto & i0 = LlvmGraphImport::create(
      graph,
      i32Type,
      pointerType,
      "i0",
      Linkage::externalLinkage,
      CallingConvention::C,
      true,
      4);

  auto & ioStateImport = GraphImport::Create(graph, ioStateType, "ioState");

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      LlvmDeltaOperation::Create(pointerType, "delta", Linkage::externalLinkage, "", true, 4));
  auto & ptrNullDeltaNode = ConstantPointerNullOperation::createNode(*deltaNode->subregion());
  auto & deltaOutput = deltaNode->finalize(ptrNullDeltaNode.output(0));

  auto lambdaNode = LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          functionType,
          "lambda",
          Linkage::externalLinkage,
          CallingConvention::C,
          {}));
  auto & ptrNullLambdaNode = ConstantPointerNullOperation::createNode(*lambdaNode->subregion());
  auto lambdaOutput = lambdaNode->finalize({ ptrNullLambdaNode.output(0) });

  auto & fnToPtrNode = CreateOpNode<FunctionToPointerOperation>({ lambdaOutput }, functionType);

  auto & oneNode = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1);
  auto & allocaNode = AllocaOperation::createNode(i32Type, *oneNode.output(0), 4);

  auto & ioBarrierNode1 =
      IOBarrierOperation::createNode(AllocaOperation::getPointerOutput(allocaNode), ioStateImport);

  auto & ioBarrierNode2 = IOBarrierOperation::createNode(i0, ioStateImport);

  auto & ioBarrierNode3 = IOBarrierOperation::createNode(deltaOutput, ioStateImport);

  auto & ioBarrierNode4 = IOBarrierOperation::createNode(*fnToPtrNode.output(0), ioStateImport);

  auto & x1 = GraphExport::Create(*ioBarrierNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*ioBarrierNode2.output(0), "x2");
  auto & x3 = GraphExport::Create(*ioBarrierNode3.output(0), "x3");
  auto & x4 = GraphExport::Create(*ioBarrierNode4.output(0), "x4");

  // Act
  rvsdg::ReduceNode<IOBarrierOperation>(
      IOBarrierOperation::normalizeDereferenceableAddressOperand,
      ioBarrierNode1);
  rvsdg::ReduceNode<IOBarrierOperation>(
      IOBarrierOperation::normalizeDereferenceableAddressOperand,
      ioBarrierNode2);
  rvsdg::ReduceNode<IOBarrierOperation>(
      IOBarrierOperation::normalizeDereferenceableAddressOperand,
      ioBarrierNode3);
  rvsdg::ReduceNode<IOBarrierOperation>(
      IOBarrierOperation::normalizeDereferenceableAddressOperand,
      ioBarrierNode4);

  // Assert
  EXPECT_EQ(x1.origin(), &AllocaOperation::getPointerOutput(allocaNode));
  EXPECT_EQ(x2.origin(), &i0);
  EXPECT_EQ(x3.origin(), &deltaOutput);
  EXPECT_EQ(x4.origin(), fnToPtrNode.output(0));
}

}
