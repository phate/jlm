/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/rvsdg/view.hpp"
#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/StoreValueForwarding.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunStoreValueForwarding(jlm::llvm::LlvmRvsdgModule & rvsdgModule)
{
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::StoreValueForwarding storeValueForwarding;
  storeValueForwarding.Run(rvsdgModule, statisticsCollector);
}

TEST(StoreValueForwardingTests, NestedAllocas)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Create a function that looks like
   * int func(io0, mem0) {
   *   p, memP0 = ALLOCA[ptr]
   *   a, memA0 = ALLOCA[int]
   *   memP1, memA1 = STORE p, a, memP0, memA0
   *   memP2, memA2 = STORE a, 20, memP1, memA1
   *   a0, memP3, memA3 = LOAD p, memP2, memA2
   *   v1, memP4, memA4 = LOAD a0, memP3, memA3
   *   return v1, io0, mem0
   * }
   *
   * After StoreValueForwarding, both ALLOCAs should be gone, and the return value should be 20
   */

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto intType = rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  const auto funcType = rvsdg::FunctionType::Create(
      { ioStateType, memoryStateType },
      { intType, ioStateType, memoryStateType });

  // Setup the function "func"
  auto & lambdaNode = *rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

  const auto io0 = lambdaNode.GetFunctionArguments()[0];
  const auto mem0 = lambdaNode.GetFunctionArguments()[1];

  auto & constantOne = rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, 1 });
  auto allocaPOutputs = AllocaOperation::create(intType, &constantOne, 4);
  auto allocaAOutputs = AllocaOperation::create(intType, &constantOne, 4);

  // Create constant 20 for the second STORE
  auto & constantTwenty = rvsdg::BitConstantOperation::create(*lambdaNode.subregion(), { 32, 20 });

  // STORE p, a, memP0, memA0
  auto & storePANode = StoreNonVolatileOperation::CreateNode(
      *allocaPOutputs[0],
      *allocaAOutputs[0],
      { allocaPOutputs[1], allocaAOutputs[1] },
      4);

  // STORE a, 20, memP1, memA1
  auto & storeA20Node = StoreNonVolatileOperation::CreateNode(
      *allocaAOutputs[0],
      constantTwenty,
      { storePANode.output(0), storePANode.output(1) },
      4);

  // LOAD p, memP2, memA2
  auto & loadPNode = LoadNonVolatileOperation::CreateNode(
      *allocaPOutputs[0],
      { storeA20Node.output(0), storeA20Node.output(1) },
      pointerType,
      8);

  // LOAD a0, memP3, memA3
  auto & loadA0Node = LoadNonVolatileOperation::CreateNode(
      *loadPNode.output(0),
      { loadPNode.output(1), loadPNode.output(2) },
      intType,
      4);

  lambdaNode.finalize({ loadA0Node.output(0), io0, mem0 });
  std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Act
  RunStoreValueForwarding(rvsdgModule);

  std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Assert

  // Both ALLOCAs should be gone, and only the LOAD of q should remain.
  size_t allocaCount = 0;
  size_t storeCount = 0;
  size_t loadCount = 0;

  for (auto & node : lambdaNode.subregion()->Nodes())
  {
    if (is<AllocaOperation>(&node))
      allocaCount++;
    else if (is<StoreOperation>(&node))
      storeCount++;
    else if (is<LoadOperation>(&node))
      loadCount++;
  }
  EXPECT_EQ(allocaCount, 0);
  EXPECT_EQ(storeCount, 0);
  EXPECT_EQ(loadCount, 0);

  // Verify that the return value is a constant 20
  const auto & result = *lambdaNode.GetFunctionResults()[0]->origin();
  const auto resultValue = llvm::tryGetConstantSignedInteger(result);
  EXPECT_EQ(resultValue, 20);
}
