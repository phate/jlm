/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/LoadChainSeparation.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
LoadNonVolatile()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & addressArgument = *lambdaNode->GetFunctionArguments()[0];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[2];

  auto & lambdaEntrySplitNode =
      LambdaEntryMemoryStateSplitOperation::CreateNode(memoryStateArgument, { 0, 1 });

  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { lambdaEntrySplitNode.output(0), lambdaEntrySplitNode.output(1) },
      valueType,
      4);

  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { loadNode1.output(1), loadNode1.output(2) },
      valueType,
      4);

  auto & loadNode3 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode2.output(2) }, valueType, 4);

  auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
      *lambdaNode->subregion(),
      { loadNode2.output(1), loadNode3.output(1) },
      { 0, 1 });

  lambdaNode->finalize({ &ioStateArgument, lambdaExitMergeNode.output(0) });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert

  // We expect the transformation to create two join nodes, one for each memory state chain.

  // Check transformation for the chain of memory state 1
  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *lambdaExitMergeNode.input(0)->origin());
    assert(joinNode && joinOperation);
    assert(joinNode->ninputs() == 2);

    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(0)->origin()) == &loadNode2);
    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(1)->origin()) == &loadNode1);

    assert(loadNode1.input(1)->origin() == lambdaEntrySplitNode.output(0));
    assert(loadNode1.input(2)->origin() == lambdaEntrySplitNode.output(1));

    assert(loadNode2.input(1)->origin() == lambdaEntrySplitNode.output(0));
    assert(loadNode2.input(2)->origin() == lambdaEntrySplitNode.output(1));
  }

  // Check transformation for the chain of memory state 2
  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *lambdaExitMergeNode.input(1)->origin());
    assert(joinNode && joinOperation);
    assert(joinNode->ninputs() == 3);

    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(0)->origin()) == &loadNode3);
    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(1)->origin()) == &loadNode2);
    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(2)->origin()) == &loadNode1);

    assert(loadNode3.input(1)->origin() == lambdaEntrySplitNode.output(1));
  }
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoadChainSeparationTests-LoadNonVolatile", LoadNonVolatile)

static void
LoadVolatile()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & addressArgument = *lambdaNode->GetFunctionArguments()[0];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[2];

  auto & loadNode1 = LoadVolatileOperation::CreateNode(
      addressArgument,
      ioStateArgument,
      { &memoryStateArgument },
      valueType,
      4);

  auto & loadNode2 = LoadVolatileOperation::CreateNode(
      addressArgument,
      LoadVolatileOperation::IOStateOutput(loadNode1),
      { &*LoadOperation::MemoryStateOutputs(loadNode1).begin() },
      valueType,
      4);

  lambdaNode->finalize({ &ioStateArgument, loadNode2.output(2) });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert

  // We expect the transformation to create a single join node with the memory state outputs of the
  // two load nodes as operands

  auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      *GetMemoryStateRegionResult(*lambdaNode).origin());
  assert(joinNode && joinOperation);
  assert(joinNode->ninputs() == 2);

  assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(0)->origin()) == &loadNode2);
  assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(1)->origin()) == &loadNode1);

  assert(loadNode1.input(2)->origin() == &memoryStateArgument);
  assert(loadNode2.input(2)->origin() == &memoryStateArgument);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoadChainSeparationTests-LoadVolatile", LoadVolatile)

static void
SingleLoad()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & addressArgument = *lambdaNode->GetFunctionArguments()[0];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[2];

  auto & loadNode =
      LoadNonVolatileOperation::CreateNode(addressArgument, { &memoryStateArgument }, valueType, 4);

  lambdaNode->finalize({ &ioStateArgument, loadNode.output(1) });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect nothing to happen as there is no chain of load nodes
  assert(
      TryGetOwnerNode<SimpleNode>(*GetMemoryStateRegionResult(*lambdaNode).origin()) == &loadNode);
  assert(LoadOperation::MemoryStateInputs(loadNode).begin()->origin() == &memoryStateArgument);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoadChainSeparationTests-SingleLoad", SingleLoad)

static void
LoadAndStore()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & addressArgument = *lambdaNode->GetFunctionArguments()[0];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[2];

  auto valueNode = TestOperation::create(lambdaNode->subregion(), {}, { valueType });

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { &memoryStateArgument }, valueType, 4);

  auto & loadNode2 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode1.output(1) }, valueType, 4);

  auto & storeNode1 = StoreNonVolatileOperation::CreateNode(
      addressArgument,
      *valueNode->output(0),
      { loadNode2.output(1) },
      4);

  auto & loadNode3 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { storeNode1.output(0) }, valueType, 4);

  auto & loadNode4 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode3.output(1) }, valueType, 4);

  auto & storeNode2 = StoreNonVolatileOperation::CreateNode(
      addressArgument,
      *valueNode->output(0),
      { loadNode4.output(1) },
      4);

  auto & loadNode5 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { storeNode2.output(0) }, valueType, 4);

  lambdaNode->finalize({ &ioStateArgument, loadNode5.output(1) });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect two join nodes to appear.
  {
    auto [joinNode, joinOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(*storeNode2.input(2)->origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }

  {
    auto [joinNode, joinOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(*storeNode1.input(2)->origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoadChainSeparationTests-LoadAndStore", LoadAndStore)

static void
GammaWithOnlyLoads()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto valueType = ValueType::Create();
  const auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create(
      { controlType, pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & controlArgument = *lambdaNode->GetFunctionArguments()[0];
  auto & addressArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[2];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[3];

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { &memoryStateArgument }, valueType, 4);

  auto & loadNode2 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode1.output(1) }, valueType, 4);

  auto gammaNode = GammaNode::create(&controlArgument, 2);
  auto addressEntryVar = gammaNode->AddEntryVar(&addressArgument);
  auto memoryStateEntryVar = gammaNode->AddEntryVar(loadNode2.output(1));

  // subregion 0
  auto & loadNode3 = LoadNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[0],
      { memoryStateEntryVar.branchArgument[0] },
      valueType,
      4);

  auto & loadNode4 = LoadNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[0],
      { loadNode3.output(1) },
      valueType,
      4);

  // subregion 1
  auto & loadNode5 = LoadNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[1],
      { memoryStateEntryVar.branchArgument[1] },
      valueType,
      4);

  auto memoryStateExitVar = gammaNode->AddExitVar({ loadNode4.output(1), loadNode5.output(1) });

  auto & loadNode6 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { memoryStateExitVar.output },
      valueType,
      4);

  auto & loadNode7 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode6.output(1) }, valueType, 4);

  lambdaNode->finalize({ &ioStateArgument, loadNode7.output(1) });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect three join nodes to appear
  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *GetMemoryStateRegionResult(*lambdaNode).origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }

  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *gammaNode->GetExitVars()[0].branchResult[0]->origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }

  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *gammaNode->GetEntryVars()[1].input->origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/LoadChainSeparationTests-GammaWithOnlyLoads",
    GammaWithOnlyLoads)

static void
GammaWithLoadsAndStores()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = ValueType::Create();
  const auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create(
      { controlType, pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & controlArgument = *lambdaNode->GetFunctionArguments()[0];
  auto & addressArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[2];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[3];

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { &memoryStateArgument }, valueType, 4);

  auto gammaNode = GammaNode::create(&controlArgument, 2);
  auto addressEntryVar = gammaNode->AddEntryVar(&addressArgument);
  auto memoryStateEntryVar = gammaNode->AddEntryVar(loadNode1.output(1));

  // subregion 0
  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[0],
      { memoryStateEntryVar.branchArgument[0] },
      valueType,
      4);

  auto & loadNode3 = LoadNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[0],
      { loadNode2.output(1) },
      valueType,
      4);

  // subregion 1
  auto value = TestOperation::create(gammaNode->subregion(1), {}, { valueType });
  auto & storeNode = StoreNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[1],
      *value->output(0),
      { memoryStateEntryVar.branchArgument[1] },
      4);

  auto & loadNode4 = LoadNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[1],
      { storeNode.output(0) },
      valueType,
      4);

  auto memoryStateExitVar = gammaNode->AddExitVar({ loadNode3.output(1), loadNode4.output(1) });

  auto & loadNode5 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { memoryStateExitVar.output },
      valueType,
      4);

  auto & loadNode6 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode5.output(1) }, valueType, 4);

  lambdaNode->finalize({ &ioStateArgument, loadNode6.output(1) });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect two join nodes to appear
  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *GetMemoryStateRegionResult(*lambdaNode).origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }

  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *gammaNode->GetExitVars()[0].branchResult[0]->origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/LoadChainSeparationTests-GammaWithLoadsAndStores",
    GammaWithLoadsAndStores)

static void
ThetaWithLoadsOnly()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto valueType = ValueType::Create();
  const auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create(
      { controlType, pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & addressArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[2];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[3];

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { &memoryStateArgument }, valueType, 4);

  auto & loadNode2 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode1.output(1) }, valueType, 4);

  auto thetaNode = ThetaNode::create(lambdaNode->subregion());

  auto addressLoopVar = thetaNode->AddLoopVar(&addressArgument);
  auto memoryStateLoopVar = thetaNode->AddLoopVar(loadNode2.output(1));

  auto & loadNode3 = LoadNonVolatileOperation::CreateNode(
      *addressLoopVar.pre,
      { memoryStateLoopVar.pre },
      valueType,
      4);

  auto & loadNode4 = LoadNonVolatileOperation::CreateNode(
      *addressLoopVar.pre,
      { loadNode3.output(1) },
      valueType,
      4);

  memoryStateLoopVar.post->divert_to(loadNode4.output(1));

  auto & loadNode5 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { memoryStateLoopVar.output },
      valueType,
      4);

  auto & loadNode6 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode5.output(1) }, valueType, 4);

  lambdaNode->finalize({ &ioStateArgument, loadNode6.output(1) });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect three join nodes to appear
  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *GetMemoryStateRegionResult(*lambdaNode).origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }

  {
    auto [joinNode, joinOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(*memoryStateLoopVar.post->origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }

  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *memoryStateLoopVar.input->origin());
    assert(joinOperation);
    assert(joinNode->ninputs() == 2);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/LoadChainSeparationTests-ThetaWithLoadsOnly",
    ThetaWithLoadsOnly)

static void
ExternalCall()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto valueType = ValueType::Create();
  const auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create(
      { controlType, pointerType, ioStateType, memoryStateType },
      { ioStateType, memoryStateType });
  const auto externalFunctionType =
      FunctionType::Create({ ioStateType, memoryStateType }, { ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & externalFunction = jlm::rvsdg::GraphImport::Create(rvsdg, externalFunctionType, "g");

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & addressArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[2];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[3];
  auto externalFunctionCtxVar = lambdaNode->AddContextVar(externalFunction);

  auto & lambdaEntrySplitNode =
      LambdaEntryMemoryStateSplitOperation::CreateNode(memoryStateArgument, { 0, 1 });

  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { lambdaEntrySplitNode.output(0) },
      valueType,
      4);

  auto & loadNode2 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode1.output(1) }, valueType, 4);

  auto & loadNode3 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { lambdaEntrySplitNode.output(1) },
      valueType,
      4);

  auto & loadNode4 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode3.output(1) }, valueType, 4);

  auto & callEntryMergeNode = CallEntryMemoryStateMergeOperation::CreateNode(
      *lambdaNode->subregion(),
      { loadNode2.output(1), loadNode4.output(1) },
      { 0, 1 });

  auto & callNode = CallOperation::CreateNode(
      externalFunctionCtxVar.inner,
      externalFunctionType,
      { &ioStateArgument, callEntryMergeNode.output(0) });

  auto & callExitSplitNode =
      CallExitMemoryStateSplitOperation::CreateNode(*callNode.output(1), { 0, 1 });

  auto & loadNode5 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { callExitSplitNode.output(0) },
      valueType,
      4);

  auto & loadNode6 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode5.output(1) }, valueType, 4);

  auto & loadNode7 = LoadNonVolatileOperation::CreateNode(
      addressArgument,
      { callExitSplitNode.output(1) },
      valueType,
      4);

  auto & loadNode8 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode7.output(1) }, valueType, 4);

  auto & lambdaExitMergeNode = LambdaExitMemoryStateMergeOperation::CreateNode(
      *lambdaNode->subregion(),
      { loadNode6.output(1), loadNode8.output(1) },
      { 0, 1 });

  lambdaNode->finalize({ callNode.output(0), lambdaExitMergeNode.output(0) });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect 4 MemoryStateJoinOperation nodes in the graph
  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *lambdaExitMergeNode.input(0)->origin());
    assert(joinOperation);

    assert(joinNode->input(0)->origin() == loadNode6.output(1));
    assert(loadNode6.input(1)->origin() == callExitSplitNode.output(0));

    assert(joinNode->input(1)->origin() == loadNode5.output(1));
    assert(loadNode5.input(1)->origin() == callExitSplitNode.output(0));
  }

  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *lambdaExitMergeNode.input(1)->origin());
    assert(joinOperation);

    assert(joinNode->input(0)->origin() == loadNode8.output(1));
    assert(loadNode8.input(1)->origin() == callExitSplitNode.output(1));

    assert(joinNode->input(1)->origin() == loadNode7.output(1));
    assert(loadNode7.input(1)->origin() == callExitSplitNode.output(1));
  }

  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *callEntryMergeNode.input(0)->origin());
    assert(joinOperation);

    assert(joinNode->input(0)->origin() == loadNode2.output(1));
    assert(loadNode2.input(1)->origin() == lambdaEntrySplitNode.output(0));

    assert(joinNode->input(1)->origin() == loadNode1.output(1));
    assert(loadNode1.input(1)->origin() == lambdaEntrySplitNode.output(0));
  }

  {
    auto [joinNode, joinOperation] = TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
        *callEntryMergeNode.input(1)->origin());
    assert(joinOperation);

    assert(joinNode->input(0)->origin() == loadNode4.output(1));
    assert(loadNode4.input(1)->origin() == lambdaEntrySplitNode.output(1));

    assert(joinNode->input(1)->origin() == loadNode3.output(1));
    assert(loadNode3.input(1)->origin() == lambdaEntrySplitNode.output(1));
  }
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoadChainSeparationTests-ExternalCall", ExternalCall)

static void
DeadOutputs()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto bit32Type = BitType::Create(32);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      { pointerType, valueType, ioStateType, memoryStateType },
      { valueType, ioStateType, memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto & addressArgument = *lambdaNode->GetFunctionArguments()[0];
  auto & valueArgument = *lambdaNode->GetFunctionArguments()[1];
  auto & ioStateArgument = *lambdaNode->GetFunctionArguments()[2];
  auto & memoryStateArgument = *lambdaNode->GetFunctionArguments()[3];

  auto & storeNode = StoreNonVolatileOperation::CreateNode(
      addressArgument,
      valueArgument,
      { &memoryStateArgument },
      4);

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { storeNode.output(0) }, valueType, 4);

  auto & loadNode2 =
      LoadNonVolatileOperation::CreateNode(addressArgument, { loadNode1.output(1) }, valueType, 4);

  auto undefValue = UndefValueOperation::Create(*lambdaNode->subregion(), memoryStateType);

  lambdaNode->finalize({
      loadNode2.output(0),
      &ioStateArgument,
      undefValue,
  });

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  assert(loadNode1.output(1)->IsDead());
  assert(loadNode1.input(1)->origin() == storeNode.output(0));
  assert(loadNode2.output(1)->IsDead());
  assert(loadNode2.input(1)->origin() == storeNode.output(0));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoadChainSeparationTests-DeadOutputs", DeadOutputs)
