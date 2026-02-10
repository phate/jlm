/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/StoreValueForwarding.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/UnitType.hpp>
#include <jlm/rvsdg/view.hpp>
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

  auto & constantOne = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 1);
  auto allocaPOutputs = AllocaOperation::create(intType, constantOne.output(0), 4);
  auto allocaAOutputs = AllocaOperation::create(intType, constantOne.output(0), 4);

  // Create constant 20 for the second STORE
  auto & constantTwenty = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 20);

  // STORE p, a, memP0, memA0
  auto & storePANode = StoreNonVolatileOperation::CreateNode(
      *allocaPOutputs[0],
      *allocaAOutputs[0],
      { allocaPOutputs[1], allocaAOutputs[1] },
      4);

  // STORE a, 20, memP1, memA1
  auto & storeA20Node = StoreNonVolatileOperation::CreateNode(
      *allocaAOutputs[0],
      *constantTwenty.output(0),
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

  // std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

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
  EXPECT_EQ(allocaCount, 0u);
  EXPECT_EQ(storeCount, 0u);
  EXPECT_EQ(loadCount, 0u);

  // Verify that the return value is a constant 20
  const auto & result = *lambdaNode.GetFunctionResults()[0]->origin();
  const auto resultValue = tryGetConstantSignedInteger(result);
  EXPECT_EQ(resultValue, 20);
}

TEST(StoreValueForwardingTests, GetElementPointerOffsets)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Create a function that looks like
   * int func(io0, mem0) {
   *   a, memA0 = ALLOCA[bits32], 2
   *   memA1 = STORE[bits32] a, 40, memA0
   *   b = GetElementPointer a, bits32[1]
   *   memA2 = STORE[bits32] b, 20, memA1
   *   l1, memA3 = LOAD[bits64] a , memA2
   *   l2, memA4 = LOAD[bits32] a , memA3
   *   c = GetElementPointer[byte] a, 4
   *   l3, memA5 = LOAD[bits32] c, memA4
   *   return l1, l2, l3, io0, mem0
   * }
   *
   * After StoreValueForwarding, the LOAD of l2 and l3 should be gone,
   * and be replaced by constant values 40 and 20, respectively
   */

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto bits32Type = rvsdg::BitType::Create(32);
  const auto bits64Type = rvsdg::BitType::Create(64);
  const auto byteType = rvsdg::BitType::Create(8);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  const auto funcType = rvsdg::FunctionType::Create(
      { ioStateType, memoryStateType },
      { bits64Type, bits32Type, bits32Type, ioStateType, memoryStateType });

  // Setup the function "func"
  auto & lambdaNode = *rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

  const auto io0 = lambdaNode.GetFunctionArguments()[0];
  const auto mem0 = lambdaNode.GetFunctionArguments()[1];

  auto & constantTwo = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 2);
  auto allocaAOutputs = AllocaOperation::create(bits32Type, constantTwo.output(0), 4);

  // Create constant 40 and 20 for the STOREs
  auto & constantTwenty = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 20);
  auto & constantForty = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 40);

  // STORE a, 40, memA0
  auto & storeA40Node = StoreNonVolatileOperation::CreateNode(
      *allocaAOutputs[0],
      *constantForty.output(0),
      { allocaAOutputs[1] },
      4);

  // b = GetElementPointer a, bits32[1]
  auto & constantOne = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 1);
  auto gepBOutput = GetElementPtrOperation::Create(
      allocaAOutputs[0],
      { constantOne.output(0) },
      bits32Type,
      pointerType);

  // STORE b, 20, memA1
  auto & storeB20Node = StoreNonVolatileOperation::CreateNode(
      *gepBOutput,
      *constantTwenty.output(0),
      { storeA40Node.output(0) },
      4);

  // l1, memA3 = LOAD[bits64] a, memA2
  auto & loadL1Node = LoadNonVolatileOperation::CreateNode(
      *allocaAOutputs[0],
      { storeB20Node.output(0) },
      bits64Type,
      8);

  // l2, memA4 = LOAD[bits32] a, memA3
  auto & loadL2Node = LoadNonVolatileOperation::CreateNode(
      *allocaAOutputs[0],
      { loadL1Node.output(1) },
      bits32Type,
      4);

  // c = GetElementPointer[byte] a, 4
  auto & constantFour = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 4);
  auto gepCOutput = GetElementPtrOperation::Create(
      allocaAOutputs[0],
      { constantFour.output(0) },
      byteType,
      pointerType);

  // l3, memA5 = LOAD[bits32] c, memA4
  auto & loadL3Node =
      LoadNonVolatileOperation::CreateNode(*gepCOutput, { loadL2Node.output(1) }, bits32Type, 4);

  lambdaNode.finalize(
      { loadL1Node.output(0), loadL2Node.output(0), loadL3Node.output(0), io0, mem0 });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Assert

  // The two stores should still be there, but only l1 should be the only LOAD
  size_t storeCount = 0;
  size_t loadCount = 0;

  for (auto & node : lambdaNode.subregion()->Nodes())
  {
    if (is<StoreOperation>(&node))
      storeCount++;
    else if (is<LoadOperation>(&node))
      loadCount++;
  }
  EXPECT_EQ(storeCount, 2u);
  EXPECT_EQ(loadCount, 1u);

  // Verify that the last two return values are constants 40 and 20
  const auto results = lambdaNode.GetFunctionResults();
  const auto r1 = tryGetConstantSignedInteger(*results[0]->origin());
  const auto r2 = tryGetConstantSignedInteger(*results[1]->origin());
  const auto r3 = tryGetConstantSignedInteger(*results[2]->origin());
  EXPECT_FALSE(r1.has_value());
  EXPECT_EQ(r2, 40);
  EXPECT_EQ(r3, 20);
}

TEST(StoreValueForwardingTests, RoutingIn)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Create a function that looks like
   * int func(q[ptr], io0, mem0) {
   *   mem1 = STORE[bits32] q, 40, mem0
   *   _, mem7, l3 = theta q, mem1, undef
   *     [q1, mem2, _] {
   *       pred = CTRL(0)
   *       l2, mem6 = gamma pred, q1, mem2
   *         [q2, mem3]{
   *             l1, mem4 = LOAD[bits32] q2, mem3
   *         }[l1, mem4]
   *         [q3, mem5]{
   *             l2 = IntegerConstantOperation(70)
   *         }[l2, mem5]
   *     }[pred, q1, mem6, l2]
   *   return l3, io0, mem7
   * }
   *
   * After StoreValueForwarding, the LOAD should be gone, and the 40 be routed in to the gamma
   */

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto bits32Type = rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto unitType = rvsdg::UnitType::Create();

  const auto funcType = rvsdg::FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { bits32Type, ioStateType, memoryStateType });

  // Setup the function "func"
  auto & lambdaNode = *rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

  auto & q = *lambdaNode.GetFunctionArguments()[0];
  auto & io0 = *lambdaNode.GetFunctionArguments()[1];
  auto & mem0 = *lambdaNode.GetFunctionArguments()[2];

  // Create constant 40 for the STORE
  auto & constantForty = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 40);

  // mem1 = STORE[bits32] q, 40, mem0
  auto & storeQ40Node =
      StoreNonVolatileOperation::CreateNode(q, *constantForty.output(0), { &mem0 }, 4);
  auto & mem1 = *StoreOperation::MemoryStateOutputs(storeQ40Node).begin();

  // Create theta node structure
  auto & thetaNode = *rvsdg::ThetaNode::create(lambdaNode.subregion());

  auto qLoopVar = thetaNode.AddLoopVar(&q);
  auto memLoopVar = thetaNode.AddLoopVar(&mem1);
  auto undefL = UndefValueOperation::Create(*lambdaNode.subregion(), bits32Type);
  auto lLoopVar = thetaNode.AddLoopVar(undefL);

  // Create gamma node inside theta
  auto & predicate = *thetaNode.predicate()->origin();
  auto & gammaNode = rvsdg::GammaNode::Create(predicate, 2, { unitType, unitType });

  auto qEntryVar = gammaNode.AddEntryVar(qLoopVar.pre);
  auto memEntryVar = gammaNode.AddEntryVar(memLoopVar.pre);

  // Create first gamma case: LOAD operation
  auto & loadNode = LoadNonVolatileOperation::CreateNode(
      *qEntryVar.branchArgument[0],
      { memEntryVar.branchArgument[0] },
      bits32Type,
      4);
  auto & loadedValue = LoadOperation::LoadedValueOutput(loadNode);
  auto & mem4 = *LoadOperation::MemoryStateOutputs(loadNode).begin();

  // Create second gamma case: constant 70
  auto & gammaSubregion1 = *gammaNode.subregion(1);
  auto & constantSeventy = IntegerConstantOperation::Create(gammaSubregion1, 32, 70);

  // Create gamma exit variables
  auto lExitVar = gammaNode.AddExitVar({ &loadedValue, constantSeventy.output(0) });
  auto memExitVar = gammaNode.AddExitVar({ &mem4, memEntryVar.branchArgument[1] });

  // route theta results
  memLoopVar.post->divert_to(memExitVar.output);
  lLoopVar.post->divert_to(lExitVar.output);

  // Finalize lambda node
  lambdaNode.finalize({ lLoopVar.output, &io0, memLoopVar.output });

  std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Act
  RunStoreValueForwarding(rvsdgModule);

  std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Assert

  // The result in gamma region 0 should be the constant 40
  auto & branch0Result = *lExitVar.branchResult[0]->origin();
  auto resultValue = tryGetConstantSignedInteger(branch0Result);
  EXPECT_EQ(resultValue, 40);

  // The result should be routed in from the constant 40 node
  auto & resultTraced = jlm::llvm::traceOutput(branch0Result);
  EXPECT_EQ(&resultTraced, constantForty.output(0));
}
