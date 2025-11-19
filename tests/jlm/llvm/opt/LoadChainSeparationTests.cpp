/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/LoadChainSeparation.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/graph.hpp>
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
  const auto valueType = ValueType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & iAddress = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "address");
  auto & iMemoryState1 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memoryState1");
  auto & iMemoryState2 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memoryState2");

  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(
      iAddress,
      { &iMemoryState1, &iMemoryState2 },
      valueType,
      4);

  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(
      iAddress,
      { loadNode1.output(1), loadNode1.output(2) },
      valueType,
      4);

  auto & loadNode3 =
      LoadNonVolatileOperation::CreateNode(iAddress, { loadNode2.output(2) }, valueType, 4);

  auto & xMemoryState1 = GraphExport::Create(*loadNode2.output(1), "memoryState1");
  auto & xMemoryState2 = GraphExport::Create(*loadNode3.output(1), "memoryState2");

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
    auto [joinNode, joinOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(*xMemoryState1.origin());
    assert(joinNode && joinOperation);
    assert(joinNode->ninputs() == 2);

    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(0)->origin()) == &loadNode2);
    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(1)->origin()) == &loadNode1);

    assert(loadNode1.input(1)->origin() == &iMemoryState1);
    assert(loadNode1.input(2)->origin() == &iMemoryState2);

    assert(loadNode2.input(1)->origin() == &iMemoryState1);
    assert(loadNode2.input(2)->origin() == &iMemoryState2);
  }

  // Check transformation for the chain of memory state 2
  {
    auto [joinNode, joinOperation] =
        TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(*xMemoryState2.origin());
    assert(joinNode && joinOperation);
    assert(joinNode->ninputs() == 3);

    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(0)->origin()) == &loadNode3);
    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(1)->origin()) == &loadNode2);
    assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(2)->origin()) == &loadNode1);

    assert(loadNode3.input(1)->origin() == &iMemoryState2);
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

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & iAddress = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "address");
  auto & iIOState = jlm::rvsdg::GraphImport::Create(rvsdg, ioStateType, "ioState");
  auto & iMemoryState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memoryState");

  auto & loadNode1 =
      LoadVolatileOperation::CreateNode(iAddress, iIOState, { &iMemoryState }, valueType, 4);

  auto & loadNode2 = LoadVolatileOperation::CreateNode(
      iAddress,
      LoadVolatileOperation::IOStateOutput(loadNode1),
      { &*LoadOperation::MemoryStateOutputs(loadNode1).begin() },
      valueType,
      4);

  auto & xMemoryState =
      GraphExport::Create(*LoadOperation::MemoryStateOutputs(loadNode2).begin(), "memoryState");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert

  // We expect the transformation to create a single join node with the memory state outputs of the
  // two load nodes as operands

  auto [joinNode, joinOperation] =
      TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(*xMemoryState.origin());
  assert(joinNode && joinOperation);
  assert(joinNode->ninputs() == 2);

  assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(0)->origin()) == &loadNode2);
  assert(TryGetOwnerNode<SimpleNode>(*joinNode->input(1)->origin()) == &loadNode1);

  assert(loadNode1.input(2)->origin() == &iMemoryState);
  assert(loadNode2.input(2)->origin() == &iMemoryState);
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
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = ValueType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & iAddress = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "address");
  auto & iMemoryState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memoryState");

  auto & loadNode = LoadNonVolatileOperation::CreateNode(iAddress, { &iMemoryState }, valueType, 4);

  auto & xMemoryState =
      GraphExport::Create(*LoadOperation::MemoryStateOutputs(loadNode).begin(), "memoryState");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect nothing to happen as there is no chain of load nodes
  assert(TryGetOwnerNode<SimpleNode>(*xMemoryState.origin()) == &loadNode);
  assert(LoadOperation::MemoryStateInputs(loadNode).begin()->origin() == &iMemoryState);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/LoadChainSeparationTests-SingleLoad", SingleLoad)

static void
GammaWithSingleLoadChainEnd()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = ValueType::Create();
  const auto controlType = ControlType::Create(2);

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & iAddress = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "address");
  auto & iMemoryState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memoryState");

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(iAddress, { &iMemoryState }, valueType, 4);

  auto predicate = TestOperation::create(&rvsdg.GetRootRegion(), {}, { controlType });

  auto gammaNode = GammaNode::create(predicate->output(0), 2);
  auto addressEntryVar = gammaNode->AddEntryVar(&iAddress);
  auto memoryStateEntryVar = gammaNode->AddEntryVar(loadNode1.output(1));

  // subregion 0
  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[0],
      { memoryStateEntryVar.branchArgument[0] },
      valueType,
      4);

  // subregion 1
  auto & loadNode3 = LoadNonVolatileOperation::CreateNode(
      *addressEntryVar.branchArgument[1],
      { memoryStateEntryVar.branchArgument[1] },
      valueType,
      4);

  auto memoryStateExitVar = gammaNode->AddExitVar({ loadNode2.output(1), loadNode3.output(1) });

  auto & loadNode5 =
      LoadNonVolatileOperation::CreateNode(iAddress, { memoryStateExitVar.output }, valueType, 4);

  GraphExport::Create(*loadNode5.output(1), "memoryState");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/LoadChainSeparationTests-GammaWithSingleLoadChainEnd",
    GammaWithSingleLoadChainEnd)
#if 0
static void
GammaWithMultipleLoadChainEnds()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = ValueType::Create();
  const auto controlType = ControlType::Create(2);

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & iAddress = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "address");
  auto & iMemoryState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memoryState");

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(iAddress, { &iMemoryState }, valueType, 4);

  auto predicate = TestOperation::create(&rvsdg.GetRootRegion(), {}, { controlType });

  auto gammaNode = GammaNode::create(predicate->output(0), 2);
  auto addressEntryVar = gammaNode->AddEntryVar(&iAddress);
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

  auto & loadNode5 =
      LoadNonVolatileOperation::CreateNode(iAddress, { memoryStateExitVar.output }, valueType, 4);

  auto & loadNode6 =
      LoadNonVolatileOperation::CreateNode(iAddress, { loadNode5.output(1) }, valueType, 4);

  GraphExport::Create(*loadNode6.output(1), "memoryState");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  LoadChainSeparation loadChainSeparation;
  loadChainSeparation.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/LoadChainSeparationTests-GammaWithMultipleLoadChainEnds",
    GammaWithMultipleLoadChainEnds)
#endif
