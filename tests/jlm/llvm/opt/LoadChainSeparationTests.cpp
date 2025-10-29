/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/LoadChainSeparation.hpp>
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
