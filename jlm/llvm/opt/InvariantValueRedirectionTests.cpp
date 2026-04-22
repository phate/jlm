/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunInvariantValueRedirection(jlm::llvm::LlvmRvsdgModule & rvsdgModule)
{
  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::InvariantValueRedirection::Configuration configuration;
  jlm::llvm::InvariantValueRedirection invariantValueRedirection(std::move(configuration));
  invariantValueRedirection.Run(rvsdgModule, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);
}

TEST(InvariantValueRedirectionTests, TestGamma)
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, valueType },
      { valueType, valueType });

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto c = lambdaNode->GetFunctionArguments()[0];
  auto x = lambdaNode->GetFunctionArguments()[1];
  auto y = lambdaNode->GetFunctionArguments()[2];

  auto gammaNode1 = jlm::rvsdg::GammaNode::create(c, 2);
  auto gammaInput1 = gammaNode1->AddEntryVar(c);
  auto gammaInput2 = gammaNode1->AddEntryVar(x);
  auto gammaInput3 = gammaNode1->AddEntryVar(y);

  auto gammaNode2 = jlm::rvsdg::GammaNode::create(gammaInput1.branchArgument[0], 2);
  auto gammaInput4 = gammaNode2->AddEntryVar(gammaInput2.branchArgument[0]);
  auto gammaInput5 = gammaNode2->AddEntryVar(gammaInput3.branchArgument[0]);
  gammaNode2->AddExitVar(gammaInput4.branchArgument);
  gammaNode2->AddExitVar(gammaInput5.branchArgument);

  gammaNode1->AddExitVar({ gammaNode2->output(0), gammaInput2.branchArgument[1] });
  gammaNode1->AddExitVar({ gammaNode2->output(1), gammaInput3.branchArgument[1] });

  auto lambdaOutput = lambdaNode->finalize({ gammaNode1->output(0), gammaNode1->output(1) });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  EXPECT_EQ(lambdaNode->GetFunctionResults()[0]->origin(), x);
  EXPECT_EQ(lambdaNode->GetFunctionResults()[1]->origin(), y);
}

TEST(InvariantValueRedirectionTests, TestTheta)
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = IOStateType::Create();
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, ioStateType },
      { controlType, valueType, ioStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto c = lambdaNode->GetFunctionArguments()[0];
  auto x = lambdaNode->GetFunctionArguments()[1];
  auto l = lambdaNode->GetFunctionArguments()[2];

  auto thetaNode1 = jlm::rvsdg::ThetaNode::create(lambdaNode->subregion());
  auto thetaVar1 = thetaNode1->AddLoopVar(c);
  auto thetaVar2 = thetaNode1->AddLoopVar(x);
  auto thetaVar3 = thetaNode1->AddLoopVar(l);

  auto thetaNode2 = jlm::rvsdg::ThetaNode::create(thetaNode1->subregion());
  auto thetaVar4 = thetaNode2->AddLoopVar(thetaVar1.pre);
  thetaNode2->AddLoopVar(thetaVar2.pre);
  auto thetaVar5 = thetaNode2->AddLoopVar(thetaVar3.pre);
  thetaNode2->set_predicate(thetaVar4.pre);

  thetaVar3.post->divert_to(thetaVar5.output);
  thetaNode1->set_predicate(thetaVar1.pre);

  auto lambdaOutput =
      lambdaNode->finalize({ thetaVar1.output, thetaVar2.output, thetaVar3.output });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  EXPECT_EQ(lambdaNode->GetFunctionResults()[0]->origin(), c);
  EXPECT_EQ(lambdaNode->GetFunctionResults()[1]->origin(), x);
  EXPECT_EQ(lambdaNode->GetFunctionResults()[2]->origin(), thetaVar3.output);
}

TEST(InvariantValueRedirectionTests, TestCall)
{
  // Arrange
  using namespace jlm::llvm;

  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionTypeTest1 = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, valueType, ioStateType, memoryStateType },
      { valueType, valueType, ioStateType, memoryStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  jlm::rvsdg::Output * lambdaOutputTest1 = nullptr;
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionTypeTest1, "test1", Linkage::externalLinkage));

    auto controlArgument = lambdaNode->GetFunctionArguments()[0];
    auto xArgument = lambdaNode->GetFunctionArguments()[1];
    auto yArgument = lambdaNode->GetFunctionArguments()[2];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[3];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[4];

    auto gammaNode = jlm::rvsdg::GammaNode::create(controlArgument, 2);
    auto gammaInputX = gammaNode->AddEntryVar(xArgument);
    auto gammaInputY = gammaNode->AddEntryVar(yArgument);
    auto gammaInputIOState = gammaNode->AddEntryVar(ioStateArgument);
    auto gammaInputMemoryState = gammaNode->AddEntryVar(memoryStateArgument);
    auto gammaOutputX =
        gammaNode->AddExitVar({ gammaInputY.branchArgument[0], gammaInputY.branchArgument[1] });
    auto gammaOutputY =
        gammaNode->AddExitVar({ gammaInputX.branchArgument[0], gammaInputX.branchArgument[1] });
    auto gammaOutputIOState = gammaNode->AddExitVar(
        { gammaInputIOState.branchArgument[0], gammaInputIOState.branchArgument[1] });
    auto gammaOutputMemoryState = gammaNode->AddExitVar(
        { gammaInputMemoryState.branchArgument[0], gammaInputMemoryState.branchArgument[1] });

    lambdaOutputTest1 = lambdaNode->finalize({ gammaOutputX.output,
                                               gammaOutputY.output,
                                               gammaOutputIOState.output,
                                               gammaOutputMemoryState.output });
  }

  jlm::rvsdg::Output * lambdaOutputTest2 = nullptr;
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { valueType, valueType, ioStateType, memoryStateType },
        { valueType, valueType, ioStateType, memoryStateType });

    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test2", Linkage::externalLinkage));
    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto yArgument = lambdaNode->GetFunctionArguments()[1];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[3];
    auto lambdaArgumentTest1 = lambdaNode->AddContextVar(*lambdaOutputTest1).inner;

    auto controlResult =
        &jlm::rvsdg::ControlConstantOperation::create(*lambdaNode->subregion(), 2, 0);

    auto & callNode = CallOperation::CreateNode(
        lambdaArgumentTest1,
        functionTypeTest1,
        AttributeList::createEmptyList(),
        { controlResult, xArgument, yArgument, ioStateArgument, memoryStateArgument });

    lambdaOutputTest2 = lambdaNode->finalize(outputs(&callNode));
    jlm::rvsdg::GraphExport::Create(*lambdaOutputTest2, "test2");
  }

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  auto & lambdaNode = jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaOutputTest2);
  EXPECT_EQ(lambdaNode.GetFunctionResults().size(), 4u);
  EXPECT_EQ(lambdaNode.GetFunctionResults()[0]->origin(), lambdaNode.GetFunctionArguments()[1]);
  EXPECT_EQ(lambdaNode.GetFunctionResults()[1]->origin(), lambdaNode.GetFunctionArguments()[0]);
  EXPECT_EQ(lambdaNode.GetFunctionResults()[2]->origin(), lambdaNode.GetFunctionArguments()[2]);
  EXPECT_EQ(lambdaNode.GetFunctionResults()[3]->origin(), lambdaNode.GetFunctionArguments()[3]);
}

TEST(InvariantValueRedirectionTests, TestCallWithMemoryStateNodes)
{
  // Arrange
  using namespace jlm::llvm;

  /**
   * Creates an RVSDG that looks like
   *
   * test1 = lambda
   *   [c:CTRL(2), x:ValueType, io, mem] {
   *     mem1, mem2 = LambdaEntrySplit{1, 2} mem
   *     // A gamma that routes the values through in both regions
   *     x2, mem3, mem4 = gamma c x mem1, mem2
   *       [_, x, mem1, mem2] {
   *       }[x, mem1, mem2]
   *       [_, x, mem1, mem2] {
   *       }[x, mem1, mem2]
   *     memMerged = LambdaExitMerge{1, 2} mem3, mem4
   *   } [x2, io, memMerged]
   *
   * test2 = lambda
   *   [test1 <- test1, x:ValueType, io, mem] {
   *     mem1, mem2 = LambdaEntrySplit{1, 2} mem
   *     callMerged = CallEntryMerge{1, 2} mem1, mem2
   *     c = CTRL(0)
   *     x2, io, returnMem = call test1 c x io callMerged
   *     mem3, mem4 = CallExitSplit{2, 1} returnMem
   *     memMerged = LambdaExitMerge{2, 1} mem3, mem4
   *   } [x2, io, memMerged]
   *
   * After InvariantValueRedirection, the LambdaExitMerge in test2 should be directly connected
   * to the LambdaEntrySplit in test2, with the correct memory node indices matching up.
   *
   * The test uses memory node indices 1 and 2, since 0 is reserved for the external node.
   */

  // The memory node representing external has index 0, and is avoided in this test
  EXPECT_EQ(aa::PointsToGraph::externalMemoryNode, 0);

  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto controlType = jlm::rvsdg::ControlType::Create(2);
  auto functionTypeTest1 = jlm::rvsdg::FunctionType::Create(
      { controlType, valueType, ioStateType, memoryStateType },
      { valueType, ioStateType, memoryStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  jlm::rvsdg::Output * lambdaOutputTest1 = nullptr;
  {
    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionTypeTest1, "test1", Linkage::externalLinkage));

    auto controlArgument = lambdaNode->GetFunctionArguments()[0];
    auto xArgument = lambdaNode->GetFunctionArguments()[1];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[3];

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 1, 2 });

    auto gammaNode = jlm::rvsdg::GammaNode::create(controlArgument, 2);

    auto gammaInputX = gammaNode->AddEntryVar(xArgument);
    auto gammaInputMemoryState1 = gammaNode->AddEntryVar(lambdaEntrySplitNode.output(0));
    auto gammaInputMemoryState2 = gammaNode->AddEntryVar(lambdaEntrySplitNode.output(1));

    auto gammaOutputX = gammaNode->AddExitVar(gammaInputX.branchArgument);
    auto gammaOutputMemoryState1 = gammaNode->AddExitVar(gammaInputMemoryState1.branchArgument);
    auto gammaOutputMemoryState2 = gammaNode->AddExitVar(gammaInputMemoryState2.branchArgument);

    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        { gammaOutputMemoryState1.output, gammaOutputMemoryState2.output },
        { 1, 2 });

    lambdaOutputTest1 = lambdaNode->finalize(
        { gammaOutputX.output, ioStateArgument, lambdaExitMergeNode.output(0) });
  }

  jlm::rvsdg::Output * lambdaOutputTest2 = nullptr;
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { valueType, ioStateType, memoryStateType },
        { valueType, ioStateType, memoryStateType });

    auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test2", Linkage::externalLinkage));
    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto lambdaArgumentTest1 = lambdaNode->AddContextVar(*lambdaOutputTest1).inner;

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 1, 2 });

    auto & callEntryMergeNode = CallEntryMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(&lambdaEntrySplitNode),
        { 1, 2 });

    auto controlResult =
        &jlm::rvsdg::ControlConstantOperation::create(*lambdaNode->subregion(), 2, 0);

    auto & callNode = CallOperation::CreateNode(
        lambdaArgumentTest1,
        functionTypeTest1,
        AttributeList::createEmptyList(),
        { controlResult, xArgument, ioStateArgument, callEntryMergeNode.output(0) });

    auto & callExitSplitNode = CallExitMemoryStateSplitOperation::CreateNode(
        CallOperation::GetMemoryStateOutput(callNode),
        { 2, 1 });

    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(&callExitSplitNode),
        { 2, 1 });

    lambdaOutputTest2 = lambdaNode->finalize({ callNode.output(0),
                                               &CallOperation::GetIOStateOutput(callNode),
                                               lambdaExitMergeNode.output(0) });
    jlm::rvsdg::GraphExport::Create(*lambdaOutputTest2, "test2");
  }

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  auto & lambdaNode = jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaOutputTest2);
  EXPECT_EQ(lambdaNode.GetFunctionResults().size(), 3u);
  EXPECT_EQ(lambdaNode.GetFunctionResults()[0]->origin(), lambdaNode.GetFunctionArguments()[0]);
  EXPECT_EQ(lambdaNode.GetFunctionResults()[1]->origin(), lambdaNode.GetFunctionArguments()[1]);

  auto lambdaEntrySplit = tryGetMemoryStateEntrySplit(lambdaNode);
  auto lambdaExitMerge = tryGetMemoryStateExitMerge(lambdaNode);

  EXPECT_TRUE(lambdaEntrySplit && lambdaEntrySplit->noutputs() == 2);
  EXPECT_TRUE(lambdaExitMerge && lambdaExitMerge->ninputs() == 2);
  EXPECT_EQ(lambdaExitMerge->input(0)->origin(), lambdaEntrySplit->output(1));
  EXPECT_EQ(lambdaExitMerge->input(1)->origin(), lambdaEntrySplit->output(0));
}

TEST(InvariantValueRedirectionTests, TestCallWithMissingMemoryStateNodes)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // The memory node representing external has index 0, and is avoided in this test
  EXPECT_EQ(aa::PointsToGraph::externalMemoryNode, 0);

  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = TestType::createValueType();
  auto int32Type = BitType::Create(32);
  auto functionType = FunctionType::Create(
      { valueType, ioStateType, memoryStateType },
      { int32Type, ioStateType, memoryStateType });

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  Output * lambdaOutputTest1 = nullptr;
  {
    auto lambdaNode = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[2];

    auto & zeroNode = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 0);
    auto & oneNode = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 1);
    auto allocaResults = AllocaOperation::create(valueType, oneNode.output(0), 4);

    auto & storeNode = StoreNonVolatileOperation::CreateNode(
        *allocaResults[0],
        *xArgument,
        { memoryStateArgument },
        4);

    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        { storeNode.output(0) },
        { 1 });

    lambdaOutputTest1 = lambdaNode->finalize(
        { zeroNode.output(0), ioStateArgument, lambdaExitMergeNode.output(0) });
  }

  Output * lambdaOutputTest2 = nullptr;
  {
    auto lambdaNode = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test2", Linkage::externalLinkage));
    auto xArgument = lambdaNode->GetFunctionArguments()[0];
    auto ioStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[2];
    auto lambdaArgumentTest = lambdaNode->AddContextVar(*lambdaOutputTest1).inner;

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 1 });

    auto & callEntryMergeNode = CallEntryMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(&lambdaEntrySplitNode),
        { 1 });

    auto & callNode = CallOperation::CreateNode(
        lambdaArgumentTest,
        functionType,
        AttributeList::createEmptyList(),
        { xArgument, ioStateArgument, callEntryMergeNode.output(0) });

    auto & callExitSplitNode = CallExitMemoryStateSplitOperation::CreateNode(
        CallOperation::GetMemoryStateOutput(callNode),
        { 1 });

    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(&callExitSplitNode),
        { 1 });

    lambdaOutputTest2 = lambdaNode->finalize({ callNode.output(0),
                                               &CallOperation::GetIOStateOutput(callNode),
                                               lambdaExitMergeNode.output(0) });
    GraphExport::Create(*lambdaOutputTest2, "test2");
  }

  std::cout << view(&rvsdg.GetRootRegion()) << std::flush;

  // Act
  RunInvariantValueRedirection(*rvsdgModule);
  std::cout << view(&rvsdg.GetRootRegion()) << std::flush;

  // Assert
  // Nothing should have been redirected
  const auto & lambdaNode1 = AssertGetOwnerNode<LambdaNode>(*lambdaOutputTest1);
  const auto lambdaEntrySplit1 = tryGetMemoryStateEntrySplit(lambdaNode1);
  const auto lambdaExitMerge1 = tryGetMemoryStateExitMerge(lambdaNode1);
  EXPECT_EQ(lambdaEntrySplit1, nullptr);
  EXPECT_TRUE(lambdaExitMerge1 && lambdaExitMerge1->ninputs() == 1);

  const auto & lambdaNode2 = AssertGetOwnerNode<LambdaNode>(*lambdaOutputTest2);
  const auto lambdaEntrySplit2 = tryGetMemoryStateEntrySplit(lambdaNode2);
  const auto lambdaExitMerge2 = tryGetMemoryStateExitMerge(lambdaNode2);
  EXPECT_TRUE(lambdaEntrySplit2 && lambdaEntrySplit2->noutputs() == 1);
  EXPECT_TRUE(lambdaExitMerge2 && lambdaExitMerge2->ninputs() == 1);
  const auto & [callExitSplitNode, _] =
      TryGetSimpleNodeAndOptionalOp<CallExitMemoryStateSplitOperation>(
          *lambdaExitMerge2->input(0)->origin());
  EXPECT_EQ(callExitSplitNode->noutputs(), 1u);
  const auto & [callNode, calOperation] =
      TryGetSimpleNodeAndOptionalOp<CallOperation>(*callExitSplitNode->input(0)->origin());
  EXPECT_EQ(callNode->noutputs(), 3u);
  EXPECT_EQ(callNode->ninputs(), 4u);
  const auto & memoryStateInput = CallOperation::GetMemoryStateInput(*callNode);
  const auto & [callEntryMergeNode, callEntryMergeOperation] =
      TryGetSimpleNodeAndOptionalOp<CallEntryMemoryStateMergeOperation>(*memoryStateInput.origin());
  EXPECT_EQ(callEntryMergeNode->ninputs(), 1u);
  EXPECT_EQ(callEntryMergeNode->input(0)->origin(), lambdaEntrySplit2->output(0));
}

TEST(InvariantValueRedirectionTests, TestCallWithDifferentExternalCompression)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  /**
   * This test creates a situation where a caller and callee have compressed some memory nodes
   * into the memory state belonging to the external memory.
   * The caller has compressed memory node 3 into external,
   * while the callees have compressed 2 into external.
   *
   * // callee0 does something to mem0
   * callee0 = lambda
   *   [io, mem] {
   *     mem0, mem1, mem3 = LambdaEntrySplit{0, 1, 3} mem
   *     mem00 = TestOperation mem0
   *     memMerged = LambdaExitMerge{0, 1, 3} mem00, mem1, mem3
   *   } [io, memMerged]
   *
   * // callee3 does something to mem3
   * callee3 = lambda
   *   [io, mem] {
   *     mem0, mem1, mem3 = LambdaEntrySplit{0, 1, 3} mem
   *     mem03 = TestOperation mem3
   *     memMerged = LambdaExitMerge{0, 1, 3} mem0, mem1, mem03
   *   } [io, memMerged]
   *
   * caller = lambda
   *   [io, mem] {
   *     mem0, mem1, mem2 = LambdaEntrySplit{0, 1, 2} mem
   *
   *     // calling callee0
   *     callMergedA = CallEntryMerge{0, 1, 2} mem0, mem1, mem2
   *     io, returnMemA = call callee0 io callMergedA
   *     mem00, mem01, mem02 = CallExitSplit{0, 1, 2} returnMemA
   *
   *     // calling callee3
   *     callMergedB = CallEntryMerge{0, 1, 2} mem00, mem01, mem02
   *     io, returnMemB = call callee3 io callMergedB
   *     mem10, mem11, mem12 = CallExitSplit{0, 1, 2} returnMemB
   *
   *     memMerged = LambdaExitMerge{0, 1, 2} mem10, mem11, mem12
   *   } [io, memMerged]
   *
   * After InvariantValueRedirection, memory node 1 should be routed around both calls.
   * Memory node 2 should only be routed around the call to callee3.
   * Memory node 0 should not be routed around anything.
   */

  // The memory node representing external has index 0
  EXPECT_EQ(aa::PointsToGraph::externalMemoryNode, 0);

  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto functionType =
      FunctionType::Create({ ioStateType, memoryStateType }, { ioStateType, memoryStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  Output * callee0Output = nullptr;
  {
    auto lambdaNode = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "callee0", Linkage::externalLinkage));

    auto ioStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 0, 1, 3 });
    auto modifiedExternal = TestOperation::createNode(
        lambdaNode->subregion(),
        { lambdaEntrySplitNode.output(0) },
        { memoryStateType });
    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        { modifiedExternal->output(0),
          lambdaEntrySplitNode.output(1),
          lambdaEntrySplitNode.output(2) },
        { 0, 1, 3 });

    callee0Output = lambdaNode->finalize({ ioStateArgument, lambdaExitMergeNode.output(0) });
  }

  Output * callee3Output = nullptr;
  {
    auto lambdaNode = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "callee3", Linkage::externalLinkage));

    auto ioStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 0, 1, 3 });
    auto modifiedMemory3 = TestOperation::createNode(
        lambdaNode->subregion(),
        { lambdaEntrySplitNode.output(2) },
        { memoryStateType });
    auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        { lambdaEntrySplitNode.output(0),
          lambdaEntrySplitNode.output(1),
          modifiedMemory3->output(0) },
        { 0, 1, 3 });

    callee3Output = lambdaNode->finalize({ ioStateArgument, lambdaExitMergeNode.output(0) });
  }

  SimpleNode * lambdaEntrySplitNode = nullptr;
  SimpleNode * callEntryMergeNodeA = nullptr;
  SimpleNode * callExitSplitNodeA = nullptr;
  SimpleNode * callEntryMergeNodeB = nullptr;
  SimpleNode * callExitSplitNodeB = nullptr;
  SimpleNode * lambdaExitMergeNode = nullptr;
  {
    auto lambdaNode = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "caller", Linkage::externalLinkage));

    auto ioStateArgument = lambdaNode->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];
    auto callee0Argument = lambdaNode->AddContextVar(*callee0Output).inner;
    auto callee3Argument = lambdaNode->AddContextVar(*callee3Output).inner;

    lambdaEntrySplitNode =
        &LambdaEntryMemoryStateSplitOperation::CreateNode(*memoryStateArgument, { 0, 1, 2 });

    callEntryMergeNodeA = &CallEntryMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(lambdaEntrySplitNode),
        { 0, 1, 2 });
    auto & callNodeA = CallOperation::CreateNode(
        callee0Argument,
        functionType,
        AttributeList::createEmptyList(),
        { ioStateArgument, callEntryMergeNodeA->output(0) });
    callExitSplitNodeA = &CallExitMemoryStateSplitOperation::CreateNode(
        CallOperation::GetMemoryStateOutput(callNodeA),
        { 0, 1, 2 });

    callEntryMergeNodeB = &CallEntryMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(callExitSplitNodeA),
        { 0, 1, 2 });
    auto & callNodeB = CallOperation::CreateNode(
        callee3Argument,
        functionType,
        AttributeList::createEmptyList(),
        { &CallOperation::GetIOStateOutput(callNodeA), callEntryMergeNodeB->output(0) });
    callExitSplitNodeB = &CallExitMemoryStateSplitOperation::CreateNode(
        CallOperation::GetMemoryStateOutput(callNodeB),
        { 0, 1, 2 });

    lambdaExitMergeNode = &LambdaExitMemoryStateMergeOperation::CreateNode(
        *lambdaNode->subregion(),
        outputs(callExitSplitNodeB),
        { 0, 1, 2 });

    lambdaNode->finalize(
        { &CallOperation::GetIOStateOutput(callNodeB), lambdaExitMergeNode->output(0) });
  }

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  ASSERT_EQ(lambdaEntrySplitNode->noutputs(), 3);
  ASSERT_EQ(callEntryMergeNodeA->ninputs(), 3);
  ASSERT_EQ(callExitSplitNodeA->noutputs(), 3);
  ASSERT_EQ(callEntryMergeNodeB->ninputs(), 3);
  ASSERT_EQ(callExitSplitNodeB->noutputs(), 3);
  ASSERT_EQ(lambdaExitMergeNode->ninputs(), 3);

  // the memory state edge representing the external node has not been re-routed around anything
  EXPECT_EQ(callEntryMergeNodeA->input(0)->origin(), lambdaEntrySplitNode->output(0));
  EXPECT_EQ(callEntryMergeNodeB->input(0)->origin(), callExitSplitNodeA->output(0));
  EXPECT_EQ(lambdaExitMergeNode->input(0)->origin(), callExitSplitNodeB->output(0));

  // the memory state edge representing memory node 1 has been re-routed to the entry
  EXPECT_EQ(callEntryMergeNodeA->input(1)->origin(), lambdaEntrySplitNode->output(1));
  EXPECT_EQ(callEntryMergeNodeB->input(1)->origin(), lambdaEntrySplitNode->output(1));
  EXPECT_EQ(lambdaExitMergeNode->input(1)->origin(), lambdaEntrySplitNode->output(1));

  // the memory state edge representing memory node 2 only been re-routed around callee3
  EXPECT_EQ(callEntryMergeNodeA->input(2)->origin(), lambdaEntrySplitNode->output(2));
  EXPECT_EQ(callEntryMergeNodeB->input(2)->origin(), callExitSplitNodeA->output(2));
  EXPECT_EQ(lambdaExitMergeNode->input(2)->origin(), callExitSplitNodeA->output(2));
}

TEST(InvariantValueRedirectionTests, TestLambdaCallArgumentMismatch)
{
  // Arrange
  jlm::llvm::LambdaCallArgumentMismatch test;

  // Act
  RunInvariantValueRedirection(test.module());

  // Assert
  auto & callNode = test.GetCall();
  auto & lambdaNode = test.GetLambdaMain();

  EXPECT_EQ(lambdaNode.GetFunctionResults().size(), 3u);
  EXPECT_EQ(lambdaNode.GetFunctionResults().size(), callNode.noutputs());
  EXPECT_EQ(lambdaNode.GetFunctionResults()[0]->origin(), callNode.output(0));
  EXPECT_EQ(lambdaNode.GetFunctionResults()[1]->origin(), callNode.output(1));
  EXPECT_EQ(lambdaNode.GetFunctionResults()[2]->origin(), callNode.output(2));
}

TEST(InvariantValueRedirectionTests, testThetaGammaRedirection)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto valueType = TestType::createValueType();
  auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create({ valueType, valueType }, { valueType });

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto functionArgument0 = lambdaNode->GetFunctionArguments()[0];
  auto functionArgument1 = lambdaNode->GetFunctionArguments()[1];

  auto thetaNode = ThetaNode::create(lambdaNode->subregion());
  auto loopVar0 = thetaNode->AddLoopVar(functionArgument0);
  auto loopVar1 = thetaNode->AddLoopVar(functionArgument1);

  auto dummyNodeTheta = TestOperation::createNode(thetaNode->subregion(), {}, { valueType });

  auto predicate =
      TestOperation::createNode(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode = GammaNode::create(predicate, 2);
  auto entryVar0 = gammaNode->AddEntryVar(loopVar0.pre);
  auto entryVar1 = gammaNode->AddEntryVar(dummyNodeTheta->output(0));

  auto dummyNodeGamma0 = TestOperation::createNode(gammaNode->subregion(0), {}, { valueType });
  auto dummyNodeGamma1 = TestOperation::createNode(gammaNode->subregion(1), {}, { valueType });

  auto controlConstant0 =
      &ControlConstantOperation::create(*gammaNode->subregion(0), ControlValueRepresentation(0, 2));
  auto controlConstant1 =
      &ControlConstantOperation::create(*gammaNode->subregion(1), ControlValueRepresentation(1, 2));

  auto controlExitVar = gammaNode->AddExitVar({ controlConstant0, controlConstant1 });
  auto exitVar0 =
      gammaNode->AddExitVar({ dummyNodeGamma0->output(0), entryVar0.branchArgument[1] });
  auto exitVar1 =
      gammaNode->AddExitVar({ entryVar1.branchArgument[0], dummyNodeGamma1->output(0) });

  thetaNode->predicate()->divert_to(controlExitVar.output);
  loopVar0.post->divert_to(exitVar0.output);
  loopVar1.post->divert_to(exitVar1.output);

  auto lambdaOutput = lambdaNode->finalize({ loopVar1.output });

  GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  // We expect that the post value of both loop variables does not originate from the gamma any
  // longer.
  auto loopVars = thetaNode->GetLoopVars();
  EXPECT_EQ(loopVars.size(), 2u);

  // Loop variable 0 was dead after the loop, which means it is irrelevant what happens to it in
  // the last iteration of the loop. As the loop predicate originates from a control constant in
  // one of the gamma nodes' subregions, the loop variables' value is always the same as the one
  // from the gamma subregion with control constant 1 (i.e. loop repetition). This means we could
  // redirect the loop variable from the gamma to the respective entry variables' origin.
  EXPECT_EQ(loopVars[0].post->origin(), loopVars[0].pre);

  // Loop variable 1 was dead at the beginning of each loop iteration, which means it is irrelevant
  // what happens to it except in the last iteration of the loop. As the loop predicate originates
  // from a control constant in a one of the gamma nodes' subregions, the loop variables' value is
  // always the same as the one from the gamma subregion with control constant 0 (i.e. loop exit).
  // This means we could redirect the loop variable from the gamma to the respective entry
  // variables' origin.
  EXPECT_EQ(loopVars[1].post->origin(), dummyNodeTheta->output(0));
}

TEST(InvariantValueRedirectionTests, testLoadWithDeadLoadedValue)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  const auto valueType = TestType::createValueType();
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto functionType = FunctionType::Create(
      { pointerType, memoryStateType, memoryStateType },
      { memoryStateType, memoryStateType });

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto addressArgument = lambdaNode->GetFunctionArguments()[0];
  auto memoryStateArgument1 = lambdaNode->GetFunctionArguments()[1];
  auto memoryStateArgument2 = lambdaNode->GetFunctionArguments()[2];

  auto & loadNode = LoadNonVolatileOperation::CreateNode(
      *addressArgument,
      { memoryStateArgument1, memoryStateArgument2 },
      valueType,
      4);

  auto lambdaOutput = lambdaNode->finalize({ loadNode.output(1), loadNode.output(2) });

  GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  // We expect that the users of the memory state outputs of the load node were redirected to the
  // origins of the respective inputs, which in turn rendered the load node dead. Consequently, it
  // was pruned from the lambda subregion.
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 0u);
  EXPECT_EQ(lambdaNode->GetFunctionResults()[0]->origin(), memoryStateArgument1);
  EXPECT_EQ(lambdaNode->GetFunctionResults()[1]->origin(), memoryStateArgument2);
}
