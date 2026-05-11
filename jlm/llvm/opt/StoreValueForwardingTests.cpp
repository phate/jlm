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
      GetElementPtrOperation::Create(allocaAOutputs[0], { constantOne.output(0) }, bits32Type);

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
  auto gepCOutput = GetElementPtrOperation::Create(allocaAOutputs[0], { constantFour.output(0) }, byteType);

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
  EXPECT_EQ(constInteger, 20);

  // In the 1st subregion, the output should be traced back to the loop var input
  const auto & traced1stRegionOrigin = jlm::llvm::traceOutput(*exitVar.branchResult[1]->origin());
  const auto loopVar2 = thetaNode.MapPreLoopVar(traced1stRegionOrigin);
  EXPECT_EQ(loopVar.pre, loopVar2.pre);

  const auto constInputInteger = jlm::llvm::tryGetConstantSignedInteger(*loopVar.input->origin());
  EXPECT_EQ(constInputInteger, 40);
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
  EXPECT_EQ(tryGetConstantSignedInteger(*loopVar2.input->origin()), 40);

  // The value replacing l3 should come from the constant directly, not a theta output.
  const auto & addRhsOrigin = *addNode.input(1)->origin();
  EXPECT_EQ(tryGetConstantSignedInteger(addRhsOrigin), 40);
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
  EXPECT_EQ(jlm::llvm::tryGetConstantSignedInteger(*exitVar.branchResult[0]->origin()), 20);
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
  auto a2 = GetElementPtrOperation::Create(
      allocaAOutputs[0],
      { constantZero.output(0), constantTwo.output(0) },
      intArrayType);
  auto a3 = GetElementPtrOperation::Create(
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
  auto a1 = GetElementPtrOperation::Create(
      aLoopVar.pre,
      { constantOneInLoop.output(0) }, bits32Type);
  auto a22 = GetElementPtrOperation::Create(a1, { constantOneInLoop.output(0) }, bits32Type);

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
  EXPECT_EQ(jlm::llvm::tryGetConstantSignedInteger(*loadedLoopVar.input->origin()), 20);
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
