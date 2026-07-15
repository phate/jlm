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
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/UnitType.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>
#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Constants.h>

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
  EXPECT_EQ(resultValue, 20u);
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
  auto gepBOutput =
      GetElementPtrOperation::create(allocaAOutputs[0], { constantOne.output(0) }, bits32Type);

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
  auto gepCOutput =
      GetElementPtrOperation::create(allocaAOutputs[0], { constantFour.output(0) }, byteType);

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
  EXPECT_EQ(r2, 40u);
  EXPECT_EQ(r3, 20u);
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
  EXPECT_EQ(resultValue, 40u);

  // The result should be routed in from the constant 40 node
  auto & resultTraced = jlm::llvm::traceOutput(branch0Result);
  EXPECT_EQ(&resultTraced, constantForty.output(0));
}

TEST(StoreValueForwardingTests, RouteOut)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Create a function that looks like
   * int func(q[ptr], io0, mem0) {
   *   mem1 = STORE[bits32] q, 40, mem0
   *   _, mem7 = theta q, mem1
   *     [q1, mem2] {
   *       pred = CTRL(0)
   *       mem6 = gamma pred, q1, mem2
   *         [_, q2, mem3]{
   *             mem4 = STORE[bits32] q2, 20, mem3
   *         }[mem4]
   *         [_, _, mem5]{
   *             // empty region
   *         }[mem5]
   *     }[pred, q1, mem6]
   *   l1, mem8 = LOAD[bits32] q, mem7
   *   return l1, io0, mem8
   * }
   *
   * After StoreValueForwarding, the LOAD should be gone.
   * The function return value should be a loop output, which leads to a gamma output,
   * which is a constant 20 in the 0th region, and invariant in the 1st region.
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

  // Create constant 40 for the first STORE
  auto & constantForty = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 40);

  // mem1 = STORE[bits32] q, 40, mem0
  auto & storeQ40Node =
      StoreNonVolatileOperation::CreateNode(q, *constantForty.output(0), { &mem0 }, 4);
  auto & mem1 = *StoreOperation::MemoryStateOutputs(storeQ40Node).begin();

  // Create theta node structure
  auto & thetaNode = *rvsdg::ThetaNode::create(lambdaNode.subregion());

  auto qLoopVar = thetaNode.AddLoopVar(&q);
  auto memLoopVar = thetaNode.AddLoopVar(&mem1);

  // Create gamma node inside theta
  auto & predicate = *thetaNode.predicate()->origin();
  auto & gammaNode = rvsdg::GammaNode::Create(predicate, 2, { unitType, unitType });

  auto qEntryVar = gammaNode.AddEntryVar(qLoopVar.pre);
  auto memEntryVar = gammaNode.AddEntryVar(memLoopVar.pre);

  // Create first gamma case: STORE operation
  auto & gammaSubregion0 = *gammaNode.subregion(0);
  auto & constantTwenty = IntegerConstantOperation::Create(gammaSubregion0, 32, 20);
  auto & storeQ20Node = StoreNonVolatileOperation::CreateNode(
      *qEntryVar.branchArgument[0],
      *constantTwenty.output(0),
      { memEntryVar.branchArgument[0] },
      4);
  auto & mem4 = *StoreOperation::MemoryStateOutputs(storeQ20Node).begin();

  // Create gamma exit variables
  auto memExitVar = gammaNode.AddExitVar({ &mem4, memEntryVar.branchArgument[1] });

  // route theta results
  memLoopVar.post->divert_to(memExitVar.output);

  // l1, mem8 = LOAD[bits32] q, mem7
  auto & loadNode =
      LoadNonVolatileOperation::CreateNode(*qLoopVar.output, { memLoopVar.output }, bits32Type, 4);
  auto & loadedValue = LoadOperation::LoadedValueOutput(loadNode);
  auto & mem8 = *LoadOperation::MemoryStateOutputs(loadNode).begin();

  // Finalize lambda node
  lambdaNode.finalize({ &loadedValue, &io0, &mem8 });

  // std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Assert

  // The LOAD should be gone, and the result should be routed through the gamma
  const auto & resultOrigin = *lambdaNode.GetFunctionResults()[0]->origin();
  const auto loopVar = thetaNode.MapOutputLoopVar(resultOrigin);

  // Inside the theta, the loop variable should come straight from the gamma
  const auto & postOrigin = *loopVar.post->origin();
  const auto exitVar = gammaNode.MapOutputExitVar(postOrigin);
  // In the 0th subregion, the output should be a constant integer
  const auto constInteger =
      jlm::llvm::tryGetConstantSignedInteger(*exitVar.branchResult[0]->origin());
  EXPECT_EQ(constInteger, 20u);

  // In the 1st subregion, the output should be traced back to the loop var input
  const auto & traced1stRegionOrigin = jlm::llvm::traceOutput(*exitVar.branchResult[1]->origin());
  const auto loopVar2 = thetaNode.MapPreLoopVar(traced1stRegionOrigin);
  EXPECT_EQ(loopVar.pre, loopVar2.pre);

  const auto constInputInteger = jlm::llvm::tryGetConstantSignedInteger(*loopVar.input->origin());
  EXPECT_EQ(constInputInteger, 40u);
}

TEST(StoreValueForwardingTests, RouteAroundLoadLoop)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Checks that StoreValueForwarding avoids routing values through loops with no stores.
   * The inner LOAD needs to have a loop variable created to provide the value,
   * but the outer LOAD should be directly replaced by the constant 40.
   *
   * Create a function that looks like
   * int func(q[ptr], io0, mem0) {
   *   mem1 = STORE[bits32] q, 40, mem0
   *   u1 = undef[bits32]
   *   _, mem4, l2 = theta q, mem1, u1
   *     [q1, mem2, _] {
   *       pred = CTRL(0)
   *       l1, mem3 = LOAD[bits32], q1, mem2
   *     }[pred, q1, mem3, l1]
   *   l3, mem5 = LOAD[bits32] q, mem4
   *   add1 = ADD l2, l3
   *   return add1, io0, mem5
   * }
   */
  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto bits32Type = rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();

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

  // mem1 = STORE[bits32] q, 40, mem0
  auto & constantForty = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 40);
  auto & storeQ40Node =
      StoreNonVolatileOperation::CreateNode(q, *constantForty.output(0), { &mem0 }, 4);
  auto & mem1 = *StoreOperation::MemoryStateOutputs(storeQ40Node).begin();

  // _, mem4, l2 = theta q, mem1, undef
  auto & thetaNode = *rvsdg::ThetaNode::create(lambdaNode.subregion());
  auto qLoopVar = thetaNode.AddLoopVar(&q);
  auto memLoopVar = thetaNode.AddLoopVar(&mem1);
  auto undefL = UndefValueOperation::Create(*lambdaNode.subregion(), bits32Type);
  auto lLoopVar = thetaNode.AddLoopVar(undefL);

  auto & loadInLoopNode =
      LoadNonVolatileOperation::CreateNode(*qLoopVar.pre, { memLoopVar.pre }, bits32Type, 4);
  auto & l1 = LoadOperation::LoadedValueOutput(loadInLoopNode);
  auto & mem3 = *LoadOperation::MemoryStateOutputs(loadInLoopNode).begin();

  lLoopVar.post->divert_to(&l1);
  memLoopVar.post->divert_to(&mem3);

  // l3, mem5 = LOAD[bits32] q, mem4
  auto & loadAfterLoopNode =
      LoadNonVolatileOperation::CreateNode(q, { memLoopVar.output }, bits32Type, 4);
  auto & l3 = LoadOperation::LoadedValueOutput(loadAfterLoopNode);
  auto & mem5 = *LoadOperation::MemoryStateOutputs(loadAfterLoopNode).begin();

  // add1 = ADD l2, l3
  auto & addNode = rvsdg::CreateOpNode<IntegerAddOperation>({ lLoopVar.output, &l3 }, 32);
  auto & add1 = *addNode.output(0);

  // return add1, io0, mem5
  lambdaNode.finalize({ &add1, &io0, &mem5 });

  std::cout << rvsdg::view(&rvsdgModule.Rvsdg().GetRootRegion()) << std::endl;

  // Act
  RunStoreValueForwarding(rvsdgModule);

  std::cout << rvsdg::view(&rvsdgModule.Rvsdg().GetRootRegion()) << std::endl;

  // Assert

  // The value replacing l2 should lead to a loop output variable,
  // whose post origin is an invariant loop variable
  const auto & addLhsOrigin = *addNode.input(0)->origin();
  const auto loopVar1 = thetaNode.MapOutputLoopVar(addLhsOrigin);
  const auto loopVar2 = thetaNode.MapPreLoopVar(*loopVar1.post->origin());
  EXPECT_TRUE(rvsdg::ThetaLoopVarIsInvariant(loopVar2));
  EXPECT_EQ(tryGetConstantSignedInteger(*loopVar2.input->origin()), 40u);

  // The value replacing l3 should come from the constant directly, not a theta output.
  const auto & addRhsOrigin = *addNode.input(1)->origin();
  EXPECT_EQ(tryGetConstantSignedInteger(addRhsOrigin), 40u);
  EXPECT_EQ(rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(addRhsOrigin), nullptr);
}

TEST(StoreValueForwardingTests, RouteUninitialized)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Checks that StoreValueForwarding creates Undef nodes when forwarding from memory
   * that has not been stored to since it was allocated.
   *
   * Create a function that looks like
   * int func(io0, mem0) {
   *   a, mem1 = ALLOCA[bits32], 1
   *   pred = CTRL(0)
   *   mem5 = gamma prec, a, mem1
   *     [_, a1, mem2] {
   *       mem3 = STORE a1, 20, mem2
   *     }[mem3]
   *     [_, a2, mem4] {
   *       // Nothing happens in this gamma branch
   *     }[mem4]
   *   ld, mem6 = LOAD[bits32] a, mem5
   *   return ld, io0, mem0
   * }
   *
   * After StoreValueForwarding, the returned value should originate from a gamma output,
   * which provides a constant integer 20 in the left region, and an undef node in the right branch.
   */
  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto bits32Type = rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto unitType = rvsdg::UnitType::Create();

  const auto funcType = rvsdg::FunctionType::Create(
      { ioStateType, memoryStateType },
      { bits32Type, ioStateType, memoryStateType });

  auto & lambdaNode = *rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

  auto & io0 = *lambdaNode.GetFunctionArguments()[0];
  auto & mem0 = *lambdaNode.GetFunctionArguments()[1];

  // a, mem1 = ALLOCA[bits32], 1
  auto & constantOne = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 1);
  auto allocaAOutputs = AllocaOperation::create(bits32Type, constantOne.output(0), 4);

  // pred = CTRL(0)
  auto & predicate = rvsdg::ControlConstantOperation::create(*lambdaNode.subregion(), 2, 0);

  // mem5 = gamma pred, a, mem1
  auto & gammaNode = rvsdg::GammaNode::Create(predicate, 2, { unitType, unitType });
  auto aEntryVar = gammaNode.AddEntryVar(allocaAOutputs[0]);
  auto memEntryVar = gammaNode.AddEntryVar(allocaAOutputs[1]);

  // [_, a1, mem2] { mem3 = STORE a1, 20, mem2 }[mem3]
  auto & gammaSubregion0 = *gammaNode.subregion(0);
  auto & constantTwenty = IntegerConstantOperation::Create(gammaSubregion0, 32, 20);
  auto & storeA20Node = StoreNonVolatileOperation::CreateNode(
      *aEntryVar.branchArgument[0],
      *constantTwenty.output(0),
      { memEntryVar.branchArgument[0] },
      4);
  auto & mem3 = *StoreOperation::MemoryStateOutputs(storeA20Node).begin();

  // [_, a2, mem4] { }[mem4]
  auto memExitVar = gammaNode.AddExitVar({ &mem3, memEntryVar.branchArgument[1] });

  // ld, mem6 = LOAD[bits32] a, mem5
  auto & loadNode = LoadNonVolatileOperation::CreateNode(
      *allocaAOutputs[0],
      { memExitVar.output },
      bits32Type,
      4);
  auto & ld = LoadOperation::LoadedValueOutput(loadNode);

  lambdaNode.finalize({ &ld, &io0, &mem0 });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  const auto & resultOrigin = *lambdaNode.GetFunctionResults()[0]->origin();
  EXPECT_NE(rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(resultOrigin), nullptr);

  const auto exitVar = gammaNode.MapOutputExitVar(resultOrigin);
  EXPECT_EQ(jlm::llvm::tryGetConstantSignedInteger(*exitVar.branchResult[0]->origin()), 20u);
  const auto [undefNode, undefOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<UndefValueOperation>(*exitVar.branchResult[1]->origin());
  EXPECT_TRUE(undefNode && undefOperation);
}

TEST(StoreValueForwardingTests, GepInLoop)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Checks that StoreValueForwarding handles GetElementPointer operations,
   * and is able to distinguish between different offsets into the same memory region.
   *
   * Creates a function that looks like
   * int func() {
   *   int a[4]; // Alloca of type int[4]
   *
   *   int* a2 = &a[2];
   *   int* a3 = &a[3];
   *   *a2 = 20;
   *   *a3 = 30;
   *
   *   do {
   *     int loaded = *a2;
   *     int* a1 = &a[1];
   *     int* a22 = &a1[1];
   *     *a22 = loaded + 1;
   *     *a1 = 10;
   *   } while(0);
   *
   *   return *a2 + *a3;
   * }
   *
   * After StoreValueForwarding, all loads should be removed, and the load of a3
   * should not go via the theta node at all.
   */

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto bits32Type = rvsdg::BitType::Create(32);
  const auto intArrayType = ArrayType::Create(bits32Type, 4);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  const auto funcType = rvsdg::FunctionType::Create(
      { ioStateType, memoryStateType },
      { bits32Type, ioStateType, memoryStateType });

  auto & lambdaNode = *rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

  auto & io0 = *lambdaNode.GetFunctionArguments()[0];
  auto & mem0 = *lambdaNode.GetFunctionArguments()[1];

  auto & constantZero = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 0);
  auto & constantOne = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 1);
  auto & constantTwo = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 2);
  auto & constantThree = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 3);
  auto & constantTwenty = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 20);
  auto & constantThirty = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 30);

  // a, mem1 = ALLOCA[int[4]], 1
  auto allocaAOutputs = AllocaOperation::create(intArrayType, constantOne.output(0), 4);

  // a2 = &a[2], a3 = &a[3]
  auto a2 = GetElementPtrOperation::create(
      allocaAOutputs[0],
      { constantZero.output(0), constantTwo.output(0) },
      intArrayType);
  auto a3 = GetElementPtrOperation::create(
      allocaAOutputs[0],
      { constantZero.output(0), constantThree.output(0) },
      intArrayType);

  // *a2 = 20; *a3 = 30;
  auto & storeA220Node = StoreNonVolatileOperation::CreateNode(
      *a2,
      *constantTwenty.output(0),
      { allocaAOutputs[1] },
      4);
  auto & mem1 = *StoreOperation::MemoryStateOutputs(storeA220Node).begin();
  auto & storeA330Node =
      StoreNonVolatileOperation::CreateNode(*a3, *constantThirty.output(0), { &mem1 }, 4);
  auto & mem2 = *StoreOperation::MemoryStateOutputs(storeA330Node).begin();

  // do { ... } while (0)
  auto & thetaNode = *rvsdg::ThetaNode::create(lambdaNode.subregion());
  auto aLoopVar = thetaNode.AddLoopVar(allocaAOutputs[0]);
  auto a2LoopVar = thetaNode.AddLoopVar(a2);
  auto memLoopVar = thetaNode.AddLoopVar(&mem2);

  // loaded = *a2;
  auto & loadInLoopNode =
      LoadNonVolatileOperation::CreateNode(*a2LoopVar.pre, { memLoopVar.pre }, bits32Type, 4);
  auto & loadedValue = LoadOperation::LoadedValueOutput(loadInLoopNode);
  auto & mem3 = *LoadOperation::MemoryStateOutputs(loadInLoopNode).begin();

  // a1 = &a[1]; a22 = &a1[1];
  auto & constantOneInLoop = IntegerConstantOperation::Create(*thetaNode.subregion(), 32, 1);
  auto a1 =
      GetElementPtrOperation::create(aLoopVar.pre, { constantOneInLoop.output(0) }, bits32Type);
  auto a22 = GetElementPtrOperation::create(a1, { constantOneInLoop.output(0) }, bits32Type);

  // *a22 = loaded + 1;
  auto & addLoadedOneNode =
      rvsdg::CreateOpNode<IntegerAddOperation>({ &loadedValue, constantOneInLoop.output(0) }, 32);
  auto & incrementedValue = *addLoadedOneNode.output(0);
  auto & storeA22Node = StoreNonVolatileOperation::CreateNode(*a22, incrementedValue, { &mem3 }, 4);
  auto & mem4 = *StoreOperation::MemoryStateOutputs(storeA22Node).begin();

  // *a1 = 10;
  auto & constantTen = IntegerConstantOperation::Create(*thetaNode.subregion(), 32, 10);
  auto & storeA1Node =
      StoreNonVolatileOperation::CreateNode(*a1, *constantTen.output(0), { &mem4 }, 4);
  auto & mem5 = *StoreOperation::MemoryStateOutputs(storeA1Node).begin();

  memLoopVar.post->divert_to(&mem5);

  // return *a2 + *a3;
  auto & loadAfterLoopA2Node =
      LoadNonVolatileOperation::CreateNode(*a2, { memLoopVar.output }, bits32Type, 4);
  auto & loadedA2 = LoadOperation::LoadedValueOutput(loadAfterLoopA2Node);
  auto & mem6 = *LoadOperation::MemoryStateOutputs(loadAfterLoopA2Node).begin();
  auto & loadAfterLoopA3Node = LoadNonVolatileOperation::CreateNode(*a3, { &mem6 }, bits32Type, 4);
  auto & loadedA3 = LoadOperation::LoadedValueOutput(loadAfterLoopA3Node);
  auto & addResultNode = rvsdg::CreateOpNode<IntegerAddOperation>({ &loadedA2, &loadedA3 }, 32);
  auto & resultValue = *addResultNode.output(0);

  lambdaNode.finalize({ &resultValue, &io0, &mem0 });

  std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Act
  RunStoreValueForwarding(rvsdgModule);

  std::cout << rvsdg::view(&graph.GetRootRegion()) << std::endl;

  // Assert

  // Check that the load of a[2] inside the loop is replaced by a loop variable
  // which takes 20 as its initial value, and loaded + 1 as its post origin
  const auto & loadedInLoopOrigin = *addLoadedOneNode.input(0)->origin();
  const auto loadedLoopVar = thetaNode.MapPreLoopVar(loadedInLoopOrigin);
  EXPECT_EQ(jlm::llvm::tryGetConstantSignedInteger(*loadedLoopVar.input->origin()), 20u);
  EXPECT_EQ(loadedLoopVar.post->origin(), addLoadedOneNode.output(0));

  // Check that the final load of a[2] is replaced by the value of loaded + 1 in the loop
  const auto & addLhsOrigin = *addResultNode.input(0)->origin();
  const auto a2ResultLoopVar = thetaNode.MapOutputLoopVar(addLhsOrigin);
  EXPECT_EQ(a2ResultLoopVar.post->origin(), addLoadedOneNode.output(0));

  // Check that the final load of a[3] is directly attached to the constant 30,
  // and that it does not go via an invariant loop variable
  const auto & addRhsOrigin = *addResultNode.input(1)->origin();
  EXPECT_EQ(&addRhsOrigin, constantThirty.output(0));
}

TEST(StoreValueForwardingTests, LoadForwarding)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Creates an RVSDG that looks like
   *
   * lambda [p:ptr, io, mem0] {
   *   l1, mem1 = load[uint32] p, mem0
   *   l2, mem2 = load[uint32] p, mem1
   *   add0 = add l1, l2
   * } [add0, io, mem2]
   *
   * and validates that only one load remains after running StoreValueForwarding.
   */

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto bits32Type = rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  const auto funcType = rvsdg::FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { bits32Type, ioStateType, memoryStateType });

  auto & lambdaNode = *rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

  auto & p = *lambdaNode.GetFunctionArguments()[0];
  auto & io0 = *lambdaNode.GetFunctionArguments()[1];
  auto & mem0 = *lambdaNode.GetFunctionArguments()[2];

  auto & load1Node = LoadNonVolatileOperation::CreateNode(p, { &mem0 }, bits32Type, 4);
  auto & l1 = LoadOperation::LoadedValueOutput(load1Node);
  auto & mem1 = *LoadOperation::MemoryStateOutputs(load1Node).begin();

  auto & load2Node = LoadNonVolatileOperation::CreateNode(p, { &mem1 }, bits32Type, 4);
  auto & l2 = LoadOperation::LoadedValueOutput(load2Node);
  auto & mem2 = *LoadOperation::MemoryStateOutputs(load2Node).begin();

  auto & addNode = rvsdg::CreateOpNode<IntegerAddOperation>({ &l1, &l2 }, 32);
  auto & add0 = *addNode.output(0);

  lambdaNode.finalize({ &add0, &io0, &mem2 });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  size_t loadCount = 0;
  for (auto & node : lambdaNode.subregion()->Nodes())
  {
    if (is<LoadOperation>(&node))
      loadCount++;
  }
  EXPECT_EQ(loadCount, 1u);

  const auto & addLhsOrigin = jlm::llvm::traceOutput(*addNode.input(0)->origin());
  const auto & addRhsOrigin = jlm::llvm::traceOutput(*addNode.input(1)->origin());
  EXPECT_EQ(&addLhsOrigin, &l1);
  EXPECT_EQ(&addRhsOrigin, &l1);

  const auto & memoryResultOrigin =
      jlm::llvm::traceOutput(*lambdaNode.GetFunctionResults()[2]->origin());
  EXPECT_EQ(&memoryResultOrigin, &mem1);
}

TEST(StoreValueForwardingTests, LoadForwardingIntoTheta)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Creates an RVSDG that looks like
   *
   * lambda [p:ptr, io, mem0] {
   *   l1, mem1 = load[uint32] p, mem0
   *
   *   _, sum, mem4 = theta p, l1, mem1 [pInner, sumInner, mem2] {
   *     l2, mem3 = load[uint32] pInner, mem2
   *     sum2 = add sumInner, l2
   *     constant100 = IntegerConstantValue[100: uint32]
   *     compare = SignedLessThan sum2, constant100
   *     predicate = MATCH[1->1, 0] compare
   *   }[predicate, pInner, sum2, mem3]
   *
   * } [sum, io, mem4]
   *
   * and validates that the load inside the theta gets removed by forwarding.
   */

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto bits32Type = rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  const auto funcType = rvsdg::FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { bits32Type, ioStateType, memoryStateType });

  auto & lambdaNode = *rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(funcType, "func", Linkage::internalLinkage));

  auto & p = *lambdaNode.GetFunctionArguments()[0];
  auto & io0 = *lambdaNode.GetFunctionArguments()[1];
  auto & mem0 = *lambdaNode.GetFunctionArguments()[2];

  auto & load1Node = LoadNonVolatileOperation::CreateNode(p, { &mem0 }, bits32Type, 4);
  auto & l1 = LoadOperation::LoadedValueOutput(load1Node);
  auto & mem1 = *LoadOperation::MemoryStateOutputs(load1Node).begin();

  auto & thetaNode = *rvsdg::ThetaNode::create(lambdaNode.subregion());
  auto pLoopVar = thetaNode.AddLoopVar(&p);
  auto sumLoopVar = thetaNode.AddLoopVar(&l1);
  auto memLoopVar = thetaNode.AddLoopVar(&mem1);

  auto & load2Node =
      LoadNonVolatileOperation::CreateNode(*pLoopVar.pre, { memLoopVar.pre }, bits32Type, 4);
  auto & l2 = LoadOperation::LoadedValueOutput(load2Node);
  auto & mem3 = *LoadOperation::MemoryStateOutputs(load2Node).begin();

  auto & addNode = rvsdg::CreateOpNode<IntegerAddOperation>({ sumLoopVar.pre, &l2 }, 32);
  auto & sum2 = *addNode.output(0);

  auto & constant100 = IntegerConstantOperation::Create(*thetaNode.subregion(), 32, 100);
  auto & sltNode = rvsdg::CreateOpNode<IntegerSltOperation>({ &sum2, constant100.output(0) }, 32);
  const auto predicate = rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  thetaNode.set_predicate(predicate);
  sumLoopVar.post->divert_to(&sum2);
  memLoopVar.post->divert_to(&mem3);

  lambdaNode.finalize({ sumLoopVar.output, &io0, memLoopVar.output });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  size_t lambdaLoadCount = 0;
  for (auto & node : lambdaNode.subregion()->Nodes())
  {
    if (is<LoadOperation>(&node))
      lambdaLoadCount++;
  }
  EXPECT_EQ(lambdaLoadCount, 1u);

  size_t thetaLoadCount = 0;
  for (auto & node : thetaNode.subregion()->Nodes())
  {
    if (is<LoadOperation>(&node))
      thetaLoadCount++;
  }
  EXPECT_EQ(thetaLoadCount, 0u);

  const auto & addLhsOrigin = jlm::llvm::traceOutput(*addNode.input(0)->origin());
  EXPECT_EQ(&addLhsOrigin, sumLoopVar.pre);

  const auto & addRhsOrigin = *addNode.input(1)->origin();
  const auto forwardedLoopVar = thetaNode.MapPreLoopVar(addRhsOrigin);
  EXPECT_TRUE(rvsdg::ThetaLoopVarIsInvariant(forwardedLoopVar));
  EXPECT_EQ(forwardedLoopVar.input->origin(), &l1);
  EXPECT_EQ(&jlm::llvm::traceOutput(addRhsOrigin), &l1);

  const auto & memoryResultOrigin =
      jlm::llvm::traceOutput(*lambdaNode.GetFunctionResults()[2]->origin());
  EXPECT_EQ(&memoryResultOrigin, &mem1);
}

TEST(StoreValueForwardingTests, LoadForwardingFromDeltaWithIntegerConstant)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  const auto pointerType = PointerType::Create();
  const auto bits8Type = BitType::Create(8);
  const auto bits32Type = BitType::Create(32);
  const auto functionType = FunctionType::Create(
      {},
      {
          bits32Type,
      });

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(bits32Type, true, pointerType));
  auto & four = IntegerConstantOperation::Create(*deltaNode->subregion(), 32, 4);
  auto & deltaOutput = deltaNode->finalize(four.output(0));

  auto & lambdaNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "func", Linkage::internalLinkage));
  auto ctxVar = lambdaNode.AddContextVar(deltaOutput);

  auto & load32Node = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, bits32Type, 4);
  auto & load8Node = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, bits8Type, 4);

  auto & zextResult = ZExtOperation::create(32, *load8Node.output(0));
  auto & addNode = IntegerAddOperation::createNode(32, *load32Node.output(0), zextResult);

  lambdaNode.finalize({ addNode.output(0) });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  // We expect all load nodes to be forwarded
  EXPECT_FALSE(Region::ContainsNodeType<LoadNonVolatileOperation>(graph.GetRootRegion(), true));
}

TEST(StoreValueForwardingTests, LoadForwardingFromDeltaWithAggregateZeroConstant)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  const auto pointerType = PointerType::Create();
  const auto bits32Type = BitType::Create(32);
  const auto structType =
      StructType::CreateIdentified("struct", { bits32Type, pointerType }, false);
  const auto functionType = FunctionType::Create({}, { bits32Type, pointerType });

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(structType, true, pointerType));
  auto aggregateZero = ConstantAggregateZeroOperation::Create(*deltaNode->subregion(), structType);
  auto & deltaOutput = deltaNode->finalize(aggregateZero);

  auto & lambdaNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "func", Linkage::internalLinkage));
  auto ctxVar = lambdaNode.AddContextVar(deltaOutput);

  auto & zeroNode = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 0);
  auto & oneNode = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 1);

  auto & gep0Node = GetElementPtrOperation::createNode(
      *ctxVar.inner,
      { zeroNode.output(0), zeroNode.output(0) },
      structType);
  auto & load32Node = LoadNonVolatileOperation::CreateNode(*gep0Node.output(0), {}, bits32Type, 4);

  auto & gep1Node = GetElementPtrOperation::createNode(
      *ctxVar.inner,
      { zeroNode.output(0), oneNode.output(0) },
      structType);
  auto & loadPtrNode =
      LoadNonVolatileOperation::CreateNode(*gep1Node.output(0), {}, pointerType, 4);

  lambdaNode.finalize({ &LoadOperation::LoadedValueOutput(load32Node),
                        &LoadOperation::LoadedValueOutput(loadPtrNode) });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  // We expect all load nodes to be forwarded
  EXPECT_FALSE(Region::ContainsNodeType<LoadNonVolatileOperation>(graph.GetRootRegion(), true));
}

TEST(StoreValueForwardingTests, LoadForwardingFloatFromDeltaWithAggregateZeroConstant)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  /**
   * Creates RVSDG corresponding to the C code:
   *
   * struct S {
   *   float f;
   *   double d;
   * };
   *
   * static const struct S myS;
   *
   * float getFloat() {
   *   return myS.f;
   * }
   * double getDouble() {
   *   return myS.d;
   * }
   *
   * where the loads in the functions have no memory state going through them,
   * since they are loading from constant memory.
   *
   * The test checks that StoreValueForwarding is able to replace the loads in the functions
   * with floating point 0.0 constants.
   */

  // Arrange - Create a single lambda with two return values for simplicity
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  const auto pointerType = PointerType::Create();
  const auto floatType = FloatingPointType::Create(fpsize::flt);
  const auto doubleType = FloatingPointType::Create(fpsize::dbl);
  const auto structType = StructType::CreateIdentified("struct", { floatType, doubleType }, false);

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(structType, true, pointerType));
  auto aggregateZero = ConstantAggregateZeroOperation::Create(*deltaNode->subregion(), structType);
  auto & deltaOutput = deltaNode->finalize(aggregateZero);

  // function types for getFloat() and getDouble()
  const auto getFloatFunctionType = FunctionType::Create({}, { floatType });
  const auto getDoubleFunctionType = FunctionType::Create({}, { doubleType });

  // Create getFloat()
  auto & getFloatLambdaNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(getFloatFunctionType, "getFloat", Linkage::internalLinkage));
  {
    auto ctxVar = getFloatLambdaNode.AddContextVar(deltaOutput);

    // Load float field (offset 0)
    auto & zeroNode = IntegerConstantOperation::Create(*getFloatLambdaNode.subregion(), 32, 0);
    auto & gepFloatNode = GetElementPtrOperation::createNode(
        *ctxVar.inner,
        { zeroNode.output(0), zeroNode.output(0) },
        structType);
    auto & loadFloatNode =
        LoadNonVolatileOperation::CreateNode(*gepFloatNode.output(0), {}, floatType, 4);
    getFloatLambdaNode.finalize({ &LoadOperation::LoadedValueOutput(loadFloatNode) });
  }

  // Create getDouble()
  auto & getDoubleFunctionNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(getDoubleFunctionType, "getDouble", Linkage::internalLinkage));
  {
    auto ctxVar = getDoubleFunctionNode.AddContextVar(deltaOutput);

    // Load double field (offset 1)
    auto & zeroNode = IntegerConstantOperation::Create(*getDoubleFunctionNode.subregion(), 32, 0);
    auto & oneNode = IntegerConstantOperation::Create(*getDoubleFunctionNode.subregion(), 32, 1);
    auto & gepDoubleNode = GetElementPtrOperation::createNode(
        *ctxVar.inner,
        { zeroNode.output(0), oneNode.output(0) },
        structType);
    auto & loadDoubleNode =
        LoadNonVolatileOperation::CreateNode(*gepDoubleNode.output(0), {}, doubleType, 8);
    getDoubleFunctionNode.finalize({ &LoadOperation::LoadedValueOutput(loadDoubleNode) });
  }

  // std::cout << jlm::rvsdg::view(&rvsdgModule.Rvsdg().GetRootRegion()) << std::endl;

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert - Check that both loads were forwarded to float zero constants
  EXPECT_FALSE(Region::ContainsNodeType<LoadNonVolatileOperation>(graph.GetRootRegion(), true));

  {
    auto floatResult = getFloatLambdaNode.GetFunctionResults()[0];
    const auto [_, floatOp] =
        jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<ConstantFP>(*floatResult->origin());
    EXPECT_NE(floatOp, nullptr);
    EXPECT_TRUE(*floatOp->result(0) == *floatType);
    EXPECT_TRUE(floatOp->constant().isZero());
    EXPECT_EQ(&floatOp->constant().getSemantics(), &::llvm::APFloat::IEEEsingle());
  }

  {
    auto doubleResult = getDoubleFunctionNode.GetFunctionResults()[0];
    const auto [_, doubleOp] =
        jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<ConstantFP>(*doubleResult->origin());
    EXPECT_NE(doubleOp, nullptr);
    EXPECT_TRUE(*doubleOp->result(0) == *doubleType);
    EXPECT_TRUE(doubleOp->constant().isZero());
    EXPECT_EQ(&doubleOp->constant().getSemantics(), &::llvm::APFloat::IEEEdouble());
  }
}

TEST(StoreValueForwardingTests, LoadForwardingFromDeltaCtxVar)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto bits32Type = BitType::Create(32);
  const auto functionType = FunctionType::Create(
      {},
      {
          pointerType,
      });

  auto deltaNode1 = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(bits32Type, true, pointerType));
  auto & four = IntegerConstantOperation::Create(*deltaNode1->subregion(), 32, 4);
  auto & deltaOutput1 = deltaNode1->finalize(four.output(0));

  auto deltaNode2 = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(pointerType, true, pointerType));
  auto deltaCtxVar = deltaNode2->AddContextVar(deltaOutput1);
  auto & deltaOutput2 = deltaNode2->finalize(deltaCtxVar.inner);

  auto & lambdaNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "func", Linkage::internalLinkage));
  auto ctxVar = lambdaNode.AddContextVar(deltaOutput2);

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, pointerType, 4);

  lambdaNode.finalize({ &LoadOperation::LoadedValueOutput(loadNode) });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  // We expect all load nodes to be forwarded
  EXPECT_FALSE(Region::ContainsNodeType<LoadNonVolatileOperation>(graph.GetRootRegion(), true));
  // We expect that deltaOutput1 has now lambdaNode as user on top of deltaNode2.
  EXPECT_EQ(deltaOutput1.nusers(), 2u);
}

TEST(StoreValueForwardingTests, LoadForwardingFromDeltaWithConstantFP)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  const auto pointerType = PointerType::Create();
  const auto floatType = FloatingPointType::Create(fpsize::flt);
  const auto functionType = FunctionType::Create(
      {},
      {
          floatType,
      });

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(floatType, true, pointerType));
  auto & fourNode =
      ConstantFP::createNode(*deltaNode->subregion(), fpsize::flt, llvm::APFloat(4.0f));
  auto & deltaOutput = deltaNode->finalize(fourNode.output(0));

  auto & lambdaNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "func", Linkage::internalLinkage));
  auto ctxVar = lambdaNode.AddContextVar(deltaOutput);

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, floatType, 4);

  lambdaNode.finalize({ &LoadOperation::LoadedValueOutput(loadNode) });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  // We expect all load nodes to be forwarded
  EXPECT_FALSE(Region::ContainsNodeType<LoadNonVolatileOperation>(graph.GetRootRegion(), true));
}

TEST(StoreValueForwardingTests, LoadForwardingFromDeltaWithConstantPointerNull)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  const auto pointerType = PointerType::Create();
  const auto functionType = FunctionType::Create(
      {},
      {
          pointerType,
      });

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(pointerType, true, pointerType));
  auto & constantPointerNull = ConstantPointerNullOperation::createNode(*deltaNode->subregion());
  auto & deltaOutput = deltaNode->finalize(constantPointerNull.output(0));

  auto & lambdaNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "func", Linkage::internalLinkage));
  auto ctxVar = lambdaNode.AddContextVar(deltaOutput);

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, pointerType, 4);

  lambdaNode.finalize({ loadNode.output(0) });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  // We expect all load nodes to be forwarded
  EXPECT_FALSE(Region::ContainsNodeType<LoadNonVolatileOperation>(graph.GetRootRegion(), true));
}

TEST(StoreValueForwardingTests, LoadForwardingFromDeltaWithConstantDataArray)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto pointerType = PointerType::Create();
  const auto bits8Type = BitType::Create(8);
  const auto bits32Type = BitType::Create(32);
  const auto bits64Type = BitType::Create(64);
  const auto arrayType = ArrayType::Create(bits32Type, 3);
  const auto functionType = FunctionType::Create(
      {},
      {
          bits32Type,
          bits32Type,
          bits32Type,
          bits32Type,
          bits32Type,
          bits64Type,
      });

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(arrayType, true, pointerType));
  auto & zeroNode = IntegerConstantOperation::Create(*deltaNode->subregion(), 32, 0);
  auto & oneNode = IntegerConstantOperation::Create(*deltaNode->subregion(), 32, 1);
  auto & twoNode = IntegerConstantOperation::Create(*deltaNode->subregion(), 32, 2);
  auto constantDataArrayResult = ConstantDataArrayOperation::Create(
      { zeroNode.output(0), oneNode.output(0), twoNode.output(0) });
  auto & deltaOutput = deltaNode->finalize(constantDataArrayResult);

  auto & lambdaNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "func", Linkage::internalLinkage));
  auto ctxVar = lambdaNode.AddContextVar(deltaOutput);

  auto zero = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 0).output(0);
  auto two = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 2).output(0);
  auto four = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 4).output(0);

  auto & loadNode0 = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, bits32Type, 4);

  auto gepOutput1 = GetElementPtrOperation::create(ctxVar.inner, { zero }, bits32Type);
  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(*gepOutput1, {}, bits32Type, 4);

  auto gepOutput2 = GetElementPtrOperation::create(ctxVar.inner, { zero, zero }, bits32Type);
  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(*gepOutput2, {}, bits32Type, 4);

  auto gepOutput3 = GetElementPtrOperation::create(ctxVar.inner, { zero, two }, bits32Type);
  auto & loadNode3 = LoadNonVolatileOperation::CreateNode(*gepOutput3, {}, bits32Type, 4);

  auto gepOutput4 = GetElementPtrOperation::create(ctxVar.inner, { four }, bits8Type);
  auto & loadNode4 = LoadNonVolatileOperation::CreateNode(*gepOutput4, {}, bits32Type, 4);

  auto & loadNode5 = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, bits64Type, 4);

  lambdaNode.finalize({
      &LoadOperation::LoadedValueOutput(loadNode0),
      &LoadOperation::LoadedValueOutput(loadNode1),
      &LoadOperation::LoadedValueOutput(loadNode2),
      &LoadOperation::LoadedValueOutput(loadNode3),
      &LoadOperation::LoadedValueOutput(loadNode4),
      &LoadOperation::LoadedValueOutput(loadNode5),
  });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  auto [intNode0, intOperation0] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[0]->origin());
  EXPECT_NE(intOperation0, nullptr);
  EXPECT_EQ(intOperation0->Representation().to_uint(), 0u);

  auto [intNode1, intOperation1] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[1]->origin());
  EXPECT_NE(intOperation1, nullptr);
  EXPECT_EQ(intOperation1->Representation().to_uint(), 0u);

  auto [intNode2, intOperation2] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[2]->origin());
  EXPECT_NE(intOperation2, nullptr);
  EXPECT_EQ(intOperation2->Representation().to_uint(), 0u);

  auto [intNode3, intOperation3] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[3]->origin());
  EXPECT_NE(intOperation3, nullptr);
  EXPECT_EQ(intOperation3->Representation().to_uint(), 2u);

  auto [intNode4, intOperation4] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[4]->origin());
  EXPECT_NE(intOperation4, nullptr);
  EXPECT_EQ(intOperation4->Representation().to_uint(), 1u);

  auto [intNode5, intOperation5] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[5]->origin());
  EXPECT_NE(intOperation5, nullptr);
  EXPECT_EQ(intOperation5->Representation().to_uint(), 0x0000000100000000u);
}

TEST(StoreValueForwardingTests, LoadForwardingFromDeltaWithConstantStruct)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto pointerType = PointerType::Create();
  const auto bits8Type = BitType::Create(8);
  const auto bits32Type = BitType::Create(32);
  const auto bits64Type = BitType::Create(64);
  const auto structType = StructType::CreateIdentified(
      { bits32Type, bits32Type, bits32Type, bits32Type, pointerType },
      false);
  auto functionType1 = FunctionType::Create({}, {});
  const auto functionType2 = FunctionType::Create(
      {},
      { bits32Type, bits32Type, bits32Type, bits32Type, bits32Type, bits64Type, pointerType });

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto & i0 = GraphImport::Create(graph, functionType2, "fct");

  auto deltaNode = DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(structType, true, pointerType));
  {
    auto ctxVar = deltaNode->AddContextVar(i0);
    auto & zeroNode = IntegerConstantOperation::Create(*deltaNode->subregion(), 32, 0);
    auto & oneNode = IntegerConstantOperation::Create(*deltaNode->subregion(), 32, 1);
    auto & twoNode = IntegerConstantOperation::Create(*deltaNode->subregion(), 32, 2);
    auto & threeNode = IntegerConstantOperation::Create(*deltaNode->subregion(), 32, 3);
    auto & fnToPtrNode = FunctionToPointerOperation::createNode(*ctxVar.inner);
    auto & constantStructResult = ConstantStructOperation::Create(
        *deltaNode->subregion(),
        { zeroNode.output(0),
          oneNode.output(0),
          twoNode.output(0),
          threeNode.output(0),
          fnToPtrNode.output(0) },
        structType);
    deltaNode->finalize(&constantStructResult);
  }

  auto & lambdaNode = *LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType2, "func", Linkage::internalLinkage));
  auto ctxVar = lambdaNode.AddContextVar(deltaNode->output());

  auto zero = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 0).output(0);
  auto two = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 2).output(0);
  auto four = IntegerConstantOperation::Create(*lambdaNode.subregion(), 32, 4).output(0);

  auto & loadNode0 = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, bits32Type, 4);

  auto gepOutput1 = GetElementPtrOperation::create(ctxVar.inner, { zero }, bits32Type);
  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(*gepOutput1, {}, bits32Type, 4);

  auto gepOutput2 = GetElementPtrOperation::create(ctxVar.inner, { zero, zero }, bits32Type);
  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(*gepOutput2, {}, bits32Type, 4);

  auto gepOutput3 = GetElementPtrOperation::create(ctxVar.inner, { zero, two }, bits32Type);
  auto & loadNode3 = LoadNonVolatileOperation::CreateNode(*gepOutput3, {}, bits32Type, 4);

  auto gepOutput4 = GetElementPtrOperation::create(ctxVar.inner, { four }, bits8Type);
  auto & loadNode4 = LoadNonVolatileOperation::CreateNode(*gepOutput4, {}, bits32Type, 4);

  auto & loadNode5 = LoadNonVolatileOperation::CreateNode(*ctxVar.inner, {}, bits64Type, 4);

  auto gepOutput6 = GetElementPtrOperation::create(ctxVar.inner, { zero, four }, bits32Type);
  auto & loadNode6 = LoadNonVolatileOperation::CreateNode(*gepOutput6, {}, pointerType, 4);

  lambdaNode.finalize({ &LoadOperation::LoadedValueOutput(loadNode0),
                        &LoadOperation::LoadedValueOutput(loadNode1),
                        &LoadOperation::LoadedValueOutput(loadNode2),
                        &LoadOperation::LoadedValueOutput(loadNode3),
                        &LoadOperation::LoadedValueOutput(loadNode4),
                        &LoadOperation::LoadedValueOutput(loadNode5),
                        &LoadOperation::LoadedValueOutput(loadNode6) });

  // Act
  RunStoreValueForwarding(rvsdgModule);

  // Assert
  auto [intNode0, intOperation0] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[0]->origin());
  EXPECT_NE(intOperation0, nullptr);
  EXPECT_EQ(intOperation0->Representation().to_uint(), 0u);

  auto [intNode1, intOperation1] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[1]->origin());
  EXPECT_NE(intOperation1, nullptr);
  EXPECT_EQ(intOperation1->Representation().to_uint(), 0u);

  auto [intNode2, intOperation2] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[2]->origin());
  EXPECT_NE(intOperation2, nullptr);
  EXPECT_EQ(intOperation2->Representation().to_uint(), 0u);

  auto [intNode3, intOperation3] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[3]->origin());
  EXPECT_NE(intOperation3, nullptr);
  EXPECT_EQ(intOperation3->Representation().to_uint(), 2u);

  auto [intNode4, intOperation4] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[4]->origin());
  EXPECT_NE(intOperation4, nullptr);
  EXPECT_EQ(intOperation4->Representation().to_uint(), 1u);

  auto [intNode5, intOperation5] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
      *lambdaNode.GetFunctionResults()[5]->origin());
  EXPECT_NE(intOperation5, nullptr);
  EXPECT_EQ(intOperation5->Representation().to_uint(), 0x0000000100000000u);

  auto [fnToPtrNode, fnToPtrOperation] = TryGetSimpleNodeAndOptionalOp<FunctionToPointerOperation>(
      *lambdaNode.GetFunctionResults()[6]->origin());
  EXPECT_NE(fnToPtrOperation, nullptr);
}
