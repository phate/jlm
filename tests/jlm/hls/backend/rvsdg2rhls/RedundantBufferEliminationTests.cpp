/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/remove-redundant-buf.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

static void
BufferWithLocalLoad()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto i64Type = jlm::rvsdg::BitType::Create(64);
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importI64 = jlm::rvsdg::GraphImport::Create(rvsdg, i64Type, "i64");
  auto & importMemState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate");
  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto loadResults = LocalLoadOperation::create(importI64, { &importMemState }, importValue);
  auto bufferResults = BufferOperation::create(*loadResults[1], 4, false);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to be replaced by a passthrough BufferOperation node
  assert(rvsdg.GetRootRegion().numNodes() == 2);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-BufferWithLocalLoad",
    BufferWithLocalLoad)

static void
BufferWithLocalStore()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto i64Type = jlm::rvsdg::BitType::Create(64);
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importI64 = jlm::rvsdg::GraphImport::Create(rvsdg, i64Type, "i64");
  auto & importMemState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate");
  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto storeResults = LocalStoreOperation::create(importI64, importValue, { &importMemState });
  auto bufferResults = BufferOperation::create(*storeResults[0], 4, false);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to be replaced by a passthrough BufferOperation node
  assert(rvsdg.GetRootRegion().numNodes() == 2);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-BufferWithLocalStore",
    BufferWithLocalStore)

static void
BufferWithLoad()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importPtr = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "ptr");
  auto & importMemState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate");
  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto loadResults = LoadOperation::create(importPtr, { &importMemState }, importValue);
  auto bufferResults = BufferOperation::create(*loadResults[1], 4, false);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to be replaced by a passthrough BufferOperation node
  assert(rvsdg.GetRootRegion().numNodes() == 2);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-BufferWithLoad",
    BufferWithLoad)

static void
BufferWithStore()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importPtr = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "ptr");
  auto & importMemState0 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate0");
  auto & importMemState1 = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate1");
  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto storeResults = jlm::hls::StoreOperation::create(
      importPtr,
      importValue,
      { &importMemState0 },
      importMemState1);
  auto bufferResults = BufferOperation::create(*storeResults[0], 4, false);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to be replaced by a passthrough BufferOperation node
  assert(rvsdg.GetRootRegion().numNodes() == 2);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-BufferWithStore",
    BufferWithStore)

static void
BufferWithForkAndLocalLoad()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto i64Type = jlm::rvsdg::BitType::Create(64);
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importI64 = jlm::rvsdg::GraphImport::Create(rvsdg, i64Type, "i64");
  auto & importMemState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate");
  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto loadResults = LocalLoadOperation::create(importI64, { &importMemState }, importValue);
  auto forkResults = ForkOperation::create(2, *loadResults[1]);
  auto bufferResults = BufferOperation::create(*forkResults[0], 4, false);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to be replaced by a passthrough BufferOperation node
  assert(rvsdg.GetRootRegion().numNodes() == 3);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-BufferWithForkAndLocalLoad",
    BufferWithForkAndLocalLoad)

static void
BufferWithBranchAndLocalLoad()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto i64Type = jlm::rvsdg::BitType::Create(64);
  auto controlType = ControlType::Create(2);
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importI64 = jlm::rvsdg::GraphImport::Create(rvsdg, i64Type, "i64");
  auto & importControl = jlm::rvsdg::GraphImport::Create(rvsdg, controlType, "control");
  auto & importMemState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate");
  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto loadResults = LocalLoadOperation::create(importI64, { &importMemState }, importValue);
  auto branchResults = BranchOperation::create(importControl, *loadResults[1]);
  auto bufferResults = BufferOperation::create(*branchResults[0], 4, false);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to be replaced by a passthrough BufferOperation node
  assert(rvsdg.GetRootRegion().numNodes() == 3);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-BufferWithBranchAndLocalLoad",
    BufferWithBranchAndLocalLoad)

static void
BufferWithOtherNode()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto node =
      TestOperation::createNode(&rvsdg.GetRootRegion(), { &importValue }, { memoryStateType });
  auto bufferResults = BufferOperation::create(*node->output(0), 4, false);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to NOT have been replaced as the operand of the
  // BufferOperation node cannot be traced to a Load-/Store-/LocalLoad-/LocalStoreOperation node
  assert(rvsdg.GetRootRegion().numNodes() == 2);
  assert(x.origin() == bufferResults[0]);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(!bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-BufferWithOtherNode",
    BufferWithOtherNode)

static void
BufferWithNonMemoryStateOperand()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto i64Type = jlm::rvsdg::BitType::Create(64);
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importI64 = jlm::rvsdg::GraphImport::Create(rvsdg, i64Type, "i64");
  auto & importMemState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate");
  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto loadResults = LocalLoadOperation::create(importI64, { &importMemState }, importValue);
  auto bufferResults = BufferOperation::create(*loadResults[0], 4, false);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to NOT have been replaced as the operand of the
  // BufferOperation node is not of type llvm::MemoryStateType
  assert(rvsdg.GetRootRegion().numNodes() == 2);
  assert(x.origin() == bufferResults[0]);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(!bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-BufferWithNonMemoryStateOperand",
    BufferWithNonMemoryStateOperand)

static void
PassthroughBuffer()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto i64Type = jlm::rvsdg::BitType::Create(64);
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importI64 = jlm::rvsdg::GraphImport::Create(rvsdg, i64Type, "i64");
  auto & importMemState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "memstate");
  auto & importValue = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "value");

  auto loadResults = LocalLoadOperation::create(importI64, { &importMemState }, importValue);
  auto bufferResults = BufferOperation::create(*loadResults[1], 4, true);

  auto & x = jlm::rvsdg::GraphExport::Create(*bufferResults[0], "x");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  RedundantBufferElimination::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  // We expect the BufferOperation node to NOT have been replaced as the BufferOperation is already
  // marked as passthrough.
  assert(rvsdg.GetRootRegion().numNodes() == 2);
  assert(x.origin() == bufferResults[0]);
  auto [bufferNode, bufferOperation] = TryGetSimpleNodeAndOptionalOp<BufferOperation>(*x.origin());
  assert(bufferNode && bufferOperation);
  assert(bufferOperation->Capacity() == 4);
  assert(bufferOperation->IsPassThrough());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/RedundantBufferEliminationTests-PassthroughBuffer",
    PassthroughBuffer)
