/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

static void
OperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::ValueType::Create();
  auto pointerType = PointerType::Create();

  LoadNonVolatileOperation operation1(valueType, 2, 4);
  LoadNonVolatileOperation operation2(pointerType, 2, 4);
  LoadNonVolatileOperation operation3(valueType, 4, 4);
  LoadNonVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::TestOperation operation5({ PointerType::Create() }, { PointerType::Create() });

  // Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // loaded type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-OperationEquality",
    OperationEquality)

static void
TestCopy()
{
  using namespace jlm::llvm;

  // Arrange
  auto memoryType = MemoryStateType::Create();
  auto valueType = jlm::tests::ValueType::Create();
  auto pointerType = PointerType::Create();

  jlm::rvsdg::Graph graph;
  auto address1 = &jlm::tests::GraphImport::Create(graph, pointerType, "address1");
  auto memoryState1 = &jlm::tests::GraphImport::Create(graph, memoryType, "memoryState1");

  auto address2 = &jlm::tests::GraphImport::Create(graph, pointerType, "address2");
  auto memoryState2 = &jlm::tests::GraphImport::Create(graph, memoryType, "memoryState2");

  auto loadResults = LoadNonVolatileOperation::Create(address1, { memoryState1 }, valueType, 4);

  // Act
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadResults[0]);
  assert(is<LoadNonVolatileOperation>(node));
  auto copiedNode = node->copy(&graph.GetRootRegion(), { address2, memoryState2 });

  // Assert
  assert(node->GetOperation() == copiedNode->GetOperation());
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/LoadNonVolatileTests-Copy", TestCopy)

static void
TestLoadAllocaReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto mt = MemoryStateType::Create();
  auto bt = jlm::rvsdg::bittype::Create(32);

  jlm::rvsdg::Graph graph;
  auto size = &jlm::tests::GraphImport::Create(graph, bt, "v");

  auto alloca1 = AllocaOperation::create(bt, size, 4);
  auto alloca2 = AllocaOperation::create(bt, size, 4);
  auto mux = MemoryStateMergeOperation::Create({ alloca1[1] });
  auto & loadNode =
      LoadNonVolatileOperation::CreateNode(*alloca1[0], { alloca1[1], alloca2[1], mux }, bt, 4);

  auto & ex = GraphExport::Create(*loadNode.output(0), "l");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadAlloca,
      loadNode);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin());
  assert(is<LoadNonVolatileOperation>(node));
  assert(node->ninputs() == 3);
  assert(node->input(1)->origin() == alloca1[1]);
  assert(node->input(2)->origin() == mux);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-LoadAllocaReduction",
    TestLoadAllocaReduction)

static void
LoadMuxReduction_Success()
{
  using namespace jlm::llvm;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();
  const auto pointerType = PointerType::Create();
  const auto bitstringType = jlm::rvsdg::bittype::Create(32);

  jlm::rvsdg::Graph graph;
  const auto address = &jlm::tests::GraphImport::Create(graph, pointerType, "address");
  auto s1 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "state1");
  auto s2 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "state2");
  auto s3 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "state3");

  auto mux = MemoryStateMergeOperation::Create({ s1, s2, s3 });
  auto & loadNode = LoadNonVolatileOperation::CreateNode(*address, { mux }, bitstringType, 4);

  auto & ex1 = GraphExport::Create(*loadNode.output(0), "l");
  auto & ex2 = GraphExport::Create(*loadNode.output(1), "s");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadMux,
      loadNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  const auto reducedLoadNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex1.origin());
  assert(is<LoadNonVolatileOperation>(reducedLoadNode));
  assert(reducedLoadNode->ninputs() == 4);
  assert(reducedLoadNode->input(0)->origin() == address);
  assert(reducedLoadNode->input(1)->origin() == s1);
  assert(reducedLoadNode->input(2)->origin() == s2);
  assert(reducedLoadNode->input(3)->origin() == s3);

  const auto merge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex2.origin());
  assert(is<MemoryStateMergeOperation>(merge));
  assert(merge->ninputs() == 3);
  for (size_t n = 0; n < merge->ninputs(); n++)
  {
    const auto expectedLoadNode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*merge->input(n)->origin());
    assert(expectedLoadNode == reducedLoadNode);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-LoadMuxReduction_Success",
    LoadMuxReduction_Success)

static void
LoadMuxReduction_WrongNumberOfOperands()
{
  // Arrange
  using namespace jlm::llvm;

  const auto vt = jlm::tests::ValueType::Create();
  const auto pt = PointerType::Create();
  const auto mt = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  const auto a = &jlm::tests::GraphImport::Create(graph, pt, "a");
  const auto s1 = &jlm::tests::GraphImport::Create(graph, mt, "s1");
  const auto s2 = &jlm::tests::GraphImport::Create(graph, mt, "s2");

  auto merge = MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::Output *>{ s1, s2 });
  auto & loadNode = LoadNonVolatileOperation::CreateNode(*a, { merge, merge }, vt, 4);

  auto & ex1 = GraphExport::Create(*loadNode.output(0), "v");
  auto & ex2 = GraphExport::Create(*loadNode.output(1), "s1");
  auto & ex3 = GraphExport::Create(*loadNode.output(2), "s2");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadMux,
      loadNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  // The LoadMux reduction should not be performed, as the current implementation does not correctly
  // take care of the two identical load state operands originating from the merge node.
  assert(success == false);
  assert(loadNode.noutputs() == 3);
  assert(ex1.origin() == loadNode.output(0));
  assert(ex2.origin() == loadNode.output(1));
  assert(ex3.origin() == loadNode.output(2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-LoadMuxReduction_WrongNumberOfOperands",
    LoadMuxReduction_WrongNumberOfOperands)

static void
LoadMuxReduction_LoadWithoutStates()
{
  using namespace jlm::llvm;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();
  const auto pointerType = PointerType::Create();

  jlm::rvsdg::Graph graph;
  const auto address = &jlm::tests::GraphImport::Create(graph, pointerType, "address");

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*address, {}, valueType, 4);

  auto & ex = GraphExport::Create(*loadNode.output(0), "v");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadMux,
      loadNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  // The load node has no states. Nothing needs to be done.
  assert(success == false);
  const auto expectedLoadNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin());
  assert(expectedLoadNode == &loadNode);
  assert(expectedLoadNode->ninputs() == 1);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-LoadMuxReduction_LoadWithoutStates",
    LoadMuxReduction_LoadWithoutStates)

static void
TestDuplicateStateReduction()
{
  using namespace jlm::llvm;

  // Arrange
  const auto memoryType = MemoryStateType::Create();
  const auto valueType = jlm::tests::ValueType::Create();
  const auto pointerType = PointerType::Create();

  jlm::rvsdg::Graph graph;
  const auto a = &jlm::tests::GraphImport::Create(graph, pointerType, "a");
  auto s1 = &jlm::tests::GraphImport::Create(graph, memoryType, "s1");
  auto s2 = &jlm::tests::GraphImport::Create(graph, memoryType, "s2");
  auto s3 = &jlm::tests::GraphImport::Create(graph, memoryType, "s3");

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*a, { s1, s2, s1, s2, s3 }, valueType, 4);

  auto & exA = GraphExport::Create(*loadNode.output(0), "exA");
  auto & exS1 = GraphExport::Create(*loadNode.output(1), "exS1");
  auto & exS2 = GraphExport::Create(*loadNode.output(2), "exS2");
  auto & exS3 = GraphExport::Create(*loadNode.output(3), "exS3");
  auto & exS4 = GraphExport::Create(*loadNode.output(4), "exS4");
  auto & exS5 = GraphExport::Create(*loadNode.output(5), "exS5");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeDuplicateStates,
      loadNode);

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  const auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exA.origin());
  assert(is<LoadNonVolatileOperation>(node));
  assert(node->ninputs() == 4);  // 1 address + 3 states
  assert(node->noutputs() == 4); // 1 loaded value + 3 states

  assert(exA.origin() == node->output(0));
  assert(exS1.origin() == node->output(1));
  assert(exS2.origin() == node->output(2));
  assert(exS3.origin() == node->output(1));
  assert(exS4.origin() == node->output(2));
  assert(exS5.origin() == node->output(3));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-DuplicateStateReduction",
    TestDuplicateStateReduction)

static void
TestLoadStoreStateReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto bt = jlm::rvsdg::bittype::Create(32);

  jlm::rvsdg::Graph graph;
  auto size = &jlm::tests::GraphImport::Create(graph, bt, "v");

  auto alloca1 = AllocaOperation::create(bt, size, 4);
  auto alloca2 = AllocaOperation::create(bt, size, 4);
  auto store1 = StoreNonVolatileOperation::Create(alloca1[0], size, { alloca1[1] }, 4);
  auto store2 = StoreNonVolatileOperation::Create(alloca2[0], size, { alloca2[1] }, 4);

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(*alloca1[0], { store1[0], store2[0] }, bt, 4);
  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(*alloca1[0], { store1[0] }, bt, 8);

  auto & ex1 = GraphExport::Create(*loadNode1.output(0), "l1");
  auto & ex2 = GraphExport::Create(*loadNode2.output(0), "l2");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success1 = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadStoreState,
      loadNode1);
  const auto success2 = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadStoreState,
      loadNode2);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success1);
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex1.origin());
  assert(is<LoadNonVolatileOperation>(node));
  assert(node->ninputs() == 2);

  assert(success2 == false);
  node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex2.origin());
  assert(is<LoadNonVolatileOperation>(node));
  assert(node->ninputs() == 2);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-LoadStoreStateReduction",
    TestLoadStoreStateReduction)

static void
TestLoadStoreReduction_Success()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::ValueType::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a = &jlm::tests::GraphImport::Create(graph, pt, "address");
  auto v = &jlm::tests::GraphImport::Create(graph, vt, "value");
  auto s = &jlm::tests::GraphImport::Create(graph, mt, "state");

  auto s1 = StoreNonVolatileOperation::Create(a, v, { s }, 4)[0];
  auto & loadNode = LoadNonVolatileOperation::CreateNode(*a, { s1 }, vt, 4);

  auto & x1 = GraphExport::Create(*loadNode.output(0), "value");
  auto & x2 = GraphExport::Create(*loadNode.output(1), "state");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadStore,
      loadNode);

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  assert(graph.GetRootRegion().nnodes() == 1);
  assert(x1.origin() == v);
  assert(x2.origin() == s1);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-LoadStoreReduction_Success",
    TestLoadStoreReduction_Success)

/**
 * Tests the load-store reduction with the value type of the store being different from the
 * value type of the load.
 */
static void
LoadStoreReduction_DifferentValueOperandType()
{
  using namespace jlm::llvm;

  // Arrange
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto & address = jlm::tests::GraphImport::Create(graph, pointerType, "address");
  auto & value = jlm::tests::GraphImport::Create(graph, jlm::rvsdg::bittype::Create(32), "value");
  auto memoryState = &jlm::tests::GraphImport::Create(graph, memoryStateType, "memoryState");

  auto & storeNode = StoreNonVolatileOperation::CreateNode(address, value, { memoryState }, 4);
  auto & loadNode = LoadNonVolatileOperation::CreateNode(
      address,
      outputs(&storeNode),
      jlm::rvsdg::bittype::Create(8),
      4);

  auto & exportedValue = GraphExport::Create(*loadNode.output(0), "v");
  GraphExport::Create(*loadNode.output(1), "s");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadStore,
      loadNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success == false);

  const auto expectedLoadNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exportedValue.origin());
  assert(expectedLoadNode == &loadNode);
  assert(expectedLoadNode->ninputs() == 2);

  const auto expectedStoreNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*expectedLoadNode->input(1)->origin());
  assert(expectedStoreNode == &storeNode);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-LoadStoreReduction_DifferentValueOperandType",
    LoadStoreReduction_DifferentValueOperandType)

static void
TestLoadLoadReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::ValueType::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a1 = &jlm::tests::GraphImport::Create(graph, pt, "a1");
  auto a2 = &jlm::tests::GraphImport::Create(graph, pt, "a2");
  auto a3 = &jlm::tests::GraphImport::Create(graph, pt, "a3");
  auto a4 = &jlm::tests::GraphImport::Create(graph, pt, "a4");
  auto v1 = &jlm::tests::GraphImport::Create(graph, vt, "v1");
  auto s1 = &jlm::tests::GraphImport::Create(graph, mt, "s1");
  auto s2 = &jlm::tests::GraphImport::Create(graph, mt, "s2");

  auto st1 = StoreNonVolatileOperation::Create(a1, v1, { s1 }, 4);
  auto ld1 = LoadNonVolatileOperation::Create(a2, { s1 }, vt, 4);
  auto ld2 = LoadNonVolatileOperation::Create(a3, { s2 }, vt, 4);

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*a4, { st1[0], ld1[1], ld2[1] }, vt, 4);

  auto & x1 = GraphExport::Create(*loadNode.output(1), "s");
  auto & x2 = GraphExport::Create(*loadNode.output(2), "s");
  auto & x3 = GraphExport::Create(*loadNode.output(3), "s");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadLoadState,
      loadNode);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  assert(graph.GetRootRegion().nnodes() == 6);

  auto ld = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*x1.origin());
  assert(is<LoadNonVolatileOperation>(ld));

  auto mx1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*x2.origin());
  assert(is<MemoryStateMergeOperation>(mx1) && mx1->ninputs() == 2);
  assert(mx1->input(0)->origin() == ld1[1] || mx1->input(0)->origin() == ld->output(2));
  assert(mx1->input(1)->origin() == ld1[1] || mx1->input(1)->origin() == ld->output(2));

  auto mx2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*x3.origin());
  assert(is<MemoryStateMergeOperation>(mx2) && mx2->ninputs() == 2);
  assert(mx2->input(0)->origin() == ld2[1] || mx2->input(0)->origin() == ld->output(3));
  assert(mx2->input(1)->origin() == ld2[1] || mx2->input(1)->origin() == ld->output(3));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadNonVolatileTests-LoadLoadReduction",
    TestLoadLoadReduction)

static void
IOBarrierAllocaAddressNormalization()
{
  using namespace jlm::llvm;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto bit32Type = jlm::rvsdg::bittype::Create(32);
  const auto ioStateType = IOStateType::Create();

  jlm::rvsdg::Graph graph;
  const auto addressImport = &jlm::tests::GraphImport::Create(graph, pointerType, "address");
  const auto sizeImport = &jlm::tests::GraphImport::Create(graph, bit32Type, "value");
  auto memoryStateImport = &jlm::tests::GraphImport::Create(graph, memoryStateType, "memState");
  auto ioStateImport = &jlm::tests::GraphImport::Create(graph, ioStateType, "ioState");

  auto allocaResults = AllocaOperation::create(valueType, sizeImport, 4);
  auto & ioBarrierNode = jlm::rvsdg::CreateOpNode<IOBarrierOperation>(
      { allocaResults[0], ioStateImport },
      pointerType);

  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(
      *ioBarrierNode.output(0),
      { allocaResults[1] },
      valueType,
      4);

  auto & loadNode2 =
      LoadNonVolatileOperation::CreateNode(*addressImport, { memoryStateImport }, valueType, 4);

  auto & ex1 = GraphExport::Create(*loadNode1.output(0), "store1");
  auto & ex2 = GraphExport::Create(*loadNode2.output(0), "store2");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  const auto successLoadNode1 = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeIOBarrierAllocaAddress,
      loadNode1);

  const auto successLoadNode2 = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeIOBarrierAllocaAddress,
      loadNode2);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(successLoadNode1);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex1.origin())->input(0)->origin()
      == allocaResults[0]);

  // There is no IOBarrierOperation node as producer for the load address. We expect the
  // normalization not to trigger.
  assert(!successLoadNode2);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex2.origin())->input(0)->origin()
      == addressImport);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadTests-IOBarrierAllocaAddressNormalization",
    IOBarrierAllocaAddressNormalization)

static void
LoadVolatileOperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::ValueType::Create();
  auto pointerType = PointerType::Create();

  LoadVolatileOperation operation1(valueType, 2, 4);
  LoadVolatileOperation operation2(pointerType, 2, 4);
  LoadVolatileOperation operation3(valueType, 4, 4);
  LoadVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::TestOperation operation5({ PointerType::Create() }, { PointerType::Create() });

  // Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // loaded type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadVolatileTests-OperationEquality",
    LoadVolatileOperationEquality)

static void
OperationCopy()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::ValueType::Create();
  PointerType pointerType;

  LoadVolatileOperation operation(valueType, 2, 4);

  // Act
  auto copiedOperation = operation.copy();

  // Assert
  assert(*copiedOperation == operation);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/LoadVolatileTests-OperationCopy", OperationCopy)

static void
OperationAccessors()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::ValueType::Create();
  PointerType pointerType;

  size_t alignment = 4;
  size_t numMemoryStates = 2;
  LoadVolatileOperation operation(valueType, numMemoryStates, alignment);

  // Assert
  assert(operation.GetLoadedType() == valueType);
  assert(operation.NumMemoryStates() == numMemoryStates);
  assert(operation.GetAlignment() == alignment);
  assert(operation.narguments() == numMemoryStates + 2); // [address, ioState, memoryStates]
  assert(operation.nresults() == numMemoryStates + 2);   // [loadedValue, ioState, memoryStates]
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadVolatileTests-OperationAccessors",
    OperationAccessors)

static void
NodeCopy()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto pointerType = PointerType::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryType = MemoryStateType::Create();
  auto valueType = jlm::tests::ValueType::Create();

  jlm::rvsdg::Graph graph;
  auto & address1 = jlm::tests::GraphImport::Create(graph, pointerType, "address1");
  auto & iOState1 = jlm::tests::GraphImport::Create(graph, iOStateType, "iOState1");
  auto & memoryState1 = jlm::tests::GraphImport::Create(graph, memoryType, "memoryState1");

  auto & address2 = jlm::tests::GraphImport::Create(graph, pointerType, "address2");
  auto & iOState2 = jlm::tests::GraphImport::Create(graph, iOStateType, "iOState2");
  auto & memoryState2 = jlm::tests::GraphImport::Create(graph, memoryType, "memoryState2");

  auto & loadNode = jlm::rvsdg::CreateOpNode<LoadVolatileOperation>(
      { &address1, &iOState1, &memoryState1 },
      valueType,
      1,
      4);

  // Act
  auto copiedNode = loadNode.copy(&graph.GetRootRegion(), { &address2, &iOState2, &memoryState2 });

  // Assert
  auto copiedOperation = dynamic_cast<const LoadVolatileOperation *>(&copiedNode->GetOperation());
  assert(copiedOperation != nullptr);
  assert(LoadOperation::AddressInput(*copiedNode).origin() == &address2);
  assert(LoadVolatileOperation::IOStateInput(*copiedNode).origin() == &iOState2);
  assert(*copiedOperation->GetLoadedType() == *valueType);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/LoadVolatileTests-NodeCopy", NodeCopy)
