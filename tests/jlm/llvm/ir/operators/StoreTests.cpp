/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

static void
StoreNonVolatileOperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::ValueType::Create();
  auto pointerType = PointerType::Create();

  StoreNonVolatileOperation operation1(valueType, 2, 4);
  StoreNonVolatileOperation operation2(pointerType, 2, 4);
  StoreNonVolatileOperation operation3(valueType, 4, 4);
  StoreNonVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::TestOperation operation5({ PointerType::Create() }, { PointerType::Create() });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // stored type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreNonVolatileOperationEquality",
    StoreNonVolatileOperationEquality)

static void
StoreVolatileOperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::ValueType::Create();
  auto pointerType = PointerType::Create();

  StoreVolatileOperation operation1(valueType, 2, 4);
  StoreVolatileOperation operation2(pointerType, 2, 4);
  StoreVolatileOperation operation3(valueType, 4, 4);
  StoreVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::TestOperation operation5({ PointerType::Create() }, { PointerType::Create() });

  // Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // stored type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileOperationEquality",
    StoreVolatileOperationEquality)

static void
StoreVolatileOperationCopy()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::ValueType::Create();
  PointerType pointerType;

  StoreVolatileOperation operation(valueType, 2, 4);

  // Act
  auto copiedOperation = operation.copy();

  // Assert
  assert(*copiedOperation == operation);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileOperationCopy",
    StoreVolatileOperationCopy)

static void
StoreVolatileOperationAccessors()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::ValueType::Create();
  PointerType pointerType;

  size_t alignment = 4;
  size_t numMemoryStates = 2;
  StoreVolatileOperation operation(valueType, numMemoryStates, alignment);

  // Assert
  assert(operation.GetStoredType() == *valueType);
  assert(operation.NumMemoryStates() == numMemoryStates);
  assert(operation.GetAlignment() == alignment);
  assert(
      operation.narguments()
      == numMemoryStates + 3); // [address, storedValue, ioState, memoryStates]
  assert(operation.nresults() == numMemoryStates + 1); // [ioState, memoryStates]
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileOperationAccessors",
    StoreVolatileOperationAccessors)

static void
StoreVolatileNodeCopy()
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto memoryType = MemoryStateType::Create();
  auto valueType = jlm::tests::ValueType::Create();

  jlm::rvsdg::Graph graph;
  auto & address1 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address1");
  auto & value1 = jlm::rvsdg::GraphImport::Create(graph, valueType, "value1");
  auto & ioState1 = jlm::rvsdg::GraphImport::Create(graph, ioStateType, "ioState1");
  auto & memoryState1 = jlm::rvsdg::GraphImport::Create(graph, memoryType, "memoryState1");

  auto & address2 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address2");
  auto & value2 = jlm::rvsdg::GraphImport::Create(graph, valueType, "value2");
  auto & ioState2 = jlm::rvsdg::GraphImport::Create(graph, ioStateType, "ioState2");
  auto & memoryState2 = jlm::rvsdg::GraphImport::Create(graph, memoryType, "memoryState2");

  auto & storeNode =
      StoreVolatileOperation::CreateNode(address1, value1, ioState1, { &memoryState1 }, 4);

  // Act
  auto copiedNode =
      storeNode.copy(&graph.GetRootRegion(), { &address2, &value2, &ioState2, &memoryState2 });

  // Assert
  assert(storeNode.GetOperation() == storeNode.GetOperation());
  assert(StoreOperation::AddressInput(*copiedNode).origin() == &address2);
  assert(StoreOperation::StoredValueInput(*copiedNode).origin() == &value2);
  assert(StoreVolatileOperation::IOStateInput(*copiedNode).origin() == &ioState2);
  assert(*StoreVolatileOperation::IOStateOutput(*copiedNode).Type() == *ioStateType);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileNodeCopy",
    StoreVolatileNodeCopy)

static void
TestCopy()
{
  using namespace jlm::llvm;

  auto valueType = jlm::tests::ValueType::Create();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto address1 = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "address1");
  auto value1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "value1");
  auto memoryState1 = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "state1");

  auto address2 = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "address2");
  auto value2 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "value2");
  auto memoryState2 = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "state2");

  auto storeResults = StoreNonVolatileOperation::Create(address1, value1, { memoryState1 }, 4);

  // Act
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*storeResults[0]);
  auto copiedNode = node->copy(&graph.GetRootRegion(), { address2, value2, memoryState2 });

  // Assert
  assert(
      node->GetOperation()
      == jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(copiedNode)->GetOperation());
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/StoreTests-TestCopy", TestCopy)

static void
TestStoreMuxNormalization()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = jlm::tests::ValueType::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a = &jlm::rvsdg::GraphImport::Create(graph, pt, "a");
  auto v = &jlm::rvsdg::GraphImport::Create(graph, vt, "v");
  auto s1 = &jlm::rvsdg::GraphImport::Create(graph, mt, "s1");
  auto s2 = &jlm::rvsdg::GraphImport::Create(graph, mt, "s2");
  auto s3 = &jlm::rvsdg::GraphImport::Create(graph, mt, "s3");

  auto mux = MemoryStateMergeOperation::Create({ s1, s2, s3 });
  auto & storeNode = StoreNonVolatileOperation::CreateNode(*a, *v, { mux }, 4);

  auto & ex = GraphExport::Create(*storeNode.output(0), "s");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  auto success = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::NormalizeStoreMux,
      storeNode);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  auto muxNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin());
  assert(is<MemoryStateMergeOperation>(muxNode));
  assert(muxNode->ninputs() == 3);
  auto n0 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*muxNode->input(0)->origin());
  auto n1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*muxNode->input(1)->origin());
  auto n2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*muxNode->input(2)->origin());
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n0->GetOperation()));
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n1->GetOperation()));
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n2->GetOperation()));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestStoreMuxNormalization",
    TestStoreMuxNormalization)

static void
TestDuplicateStateReduction()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "a");
  auto v = &jlm::rvsdg::GraphImport::Create(graph, valueType, "v");
  auto s1 = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "s1");
  auto s2 = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "s2");
  auto s3 = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "s3");

  auto & storeNode = StoreNonVolatileOperation::CreateNode(*a, *v, { s1, s2, s1, s2, s3 }, 4);

  auto & exS1 = GraphExport::Create(*storeNode.output(0), "exS1");
  auto & exS2 = GraphExport::Create(*storeNode.output(1), "exS2");
  auto & exS3 = GraphExport::Create(*storeNode.output(2), "exS3");
  auto & exS4 = GraphExport::Create(*storeNode.output(3), "exS4");
  auto & exS5 = GraphExport::Create(*storeNode.output(4), "exS5");

  view(&graph.GetRootRegion(), stdout);

  // Act
  auto success = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::NormalizeDuplicateStates,
      storeNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exS1.origin());
  assert(is<StoreNonVolatileOperation>(node));
  assert(node->ninputs() == 5);
  assert(node->noutputs() == 3);
  assert(exS1.origin() == node->output(0));
  assert(exS2.origin() == node->output(1));
  assert(exS3.origin() == node->output(0));
  assert(exS4.origin() == node->output(1));
  assert(exS5.origin() == node->output(2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestDuplicateStateReduction",
    TestDuplicateStateReduction)

static void
TestStoreAllocaReduction()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = jlm::tests::ValueType::Create();
  auto mt = MemoryStateType::Create();
  auto bt = jlm::rvsdg::BitType::Create(32);

  jlm::rvsdg::Graph graph;
  auto size = &jlm::rvsdg::GraphImport::Create(graph, bt, "size");
  auto value = &jlm::rvsdg::GraphImport::Create(graph, vt, "value");
  auto s = &jlm::rvsdg::GraphImport::Create(graph, mt, "s");

  auto alloca1 = AllocaOperation::create(vt, size, 4);
  auto alloca2 = AllocaOperation::create(vt, size, 4);
  auto & storeNode1 =
      StoreNonVolatileOperation::CreateNode(*alloca1[0], *value, { alloca1[1], alloca2[1], s }, 4);
  auto & storeNode2 =
      StoreNonVolatileOperation::CreateNode(*alloca2[0], *value, outputs(&storeNode1), 4);

  GraphExport::Create(*storeNode2.output(0), "s1");
  GraphExport::Create(*storeNode2.output(1), "s2");
  GraphExport::Create(*storeNode2.output(2), "s3");

  view(&graph.GetRootRegion(), stdout);

  // Act
  auto success1 = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::NormalizeStoreAlloca,
      storeNode1);
  auto success2 = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::NormalizeStoreAlloca,
      storeNode2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success1 && success2);
  bool has_add_import = false;
  for (size_t n = 0; n < graph.GetRootRegion().nresults(); n++)
  {
    if (graph.GetRootRegion().result(n)->origin() == s)
      has_add_import = true;
  }
  assert(has_add_import);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestStoreAllocaReduction",
    TestStoreAllocaReduction)

static void
TestStoreStoreReduction()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = jlm::tests::ValueType::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a = &jlm::rvsdg::GraphImport::Create(graph, pt, "address");
  auto v1 = &jlm::rvsdg::GraphImport::Create(graph, vt, "value");
  auto v2 = &jlm::rvsdg::GraphImport::Create(graph, vt, "value");
  auto s = &jlm::rvsdg::GraphImport::Create(graph, mt, "state");

  auto & storeNode1 = StoreNonVolatileOperation::CreateNode(*a, *v1, { s }, 4);
  auto & storeNode2 = StoreNonVolatileOperation::CreateNode(*a, *v2, outputs(&storeNode1), 4);

  auto & ex = GraphExport::Create(*storeNode2.output(0), "state");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  auto success = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::NormalizeStoreStore,
      storeNode2);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  assert(graph.GetRootRegion().nnodes() == 1);
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin())->input(1)->origin() == v2);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestStoreStoreReduction",
    TestStoreStoreReduction)

static void
IOBarrierAllocaAddressNormalization()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto bit32Type = jlm::rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();

  jlm::rvsdg::Graph graph;
  const auto addressImport = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "address");
  const auto valueImport = &jlm::rvsdg::GraphImport::Create(graph, valueType, "value");
  const auto sizeImport = &jlm::rvsdg::GraphImport::Create(graph, bit32Type, "value");
  auto memoryStateImport = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memState");
  auto ioStateImport = &jlm::rvsdg::GraphImport::Create(graph, ioStateType, "ioState");

  auto allocaResults = AllocaOperation::create(valueType, sizeImport, 4);
  auto & ioBarrierNode = jlm::rvsdg::CreateOpNode<IOBarrierOperation>(
      { allocaResults[0], ioStateImport },
      pointerType);

  auto & storeNode1 = StoreNonVolatileOperation::CreateNode(
      *ioBarrierNode.output(0),
      *valueImport,
      { allocaResults[1] },
      4);

  auto & storeNode2 =
      StoreNonVolatileOperation::CreateNode(*addressImport, *valueImport, { memoryStateImport }, 4);

  auto & ex1 = GraphExport::Create(*storeNode1.output(0), "store1");
  auto & ex2 = GraphExport::Create(*storeNode2.output(0), "store2");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  const auto successStoreNode1 = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::NormalizeIOBarrierAllocaAddress,
      storeNode1);

  const auto successStoreNode2 = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::NormalizeIOBarrierAllocaAddress,
      storeNode2);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(successStoreNode1);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex1.origin())->input(0)->origin()
      == allocaResults[0]);

  // There is no IOBarrierOperation node as producer for the store address. We expect the
  // normalization not to trigger.
  assert(!successStoreNode2);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex2.origin())->input(0)->origin()
      == addressImport);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestIOBarrierAllocaAddressNormalization",
    IOBarrierAllocaAddressNormalization)

static void
IOBarrierAllocaAddressNormalization_Gamma()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();
  const auto pointerType = PointerType::Create();
  const auto bit32Type = jlm::rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto controlTye = ControlType::Create(2);

  Graph graph;
  const auto valueImport = &jlm::rvsdg::GraphImport::Create(graph, valueType, "value");
  const auto sizeImport = &jlm::rvsdg::GraphImport::Create(graph, bit32Type, "value");
  auto ioStateImport = &jlm::rvsdg::GraphImport::Create(graph, ioStateType, "ioState");
  const auto controlImport = &jlm::rvsdg::GraphImport::Create(graph, controlTye, "control");

  auto allocaResults = AllocaOperation::create(valueType, sizeImport, 4);

  auto gammaNode = GammaNode::create(controlImport, 2);
  auto addressEntryVar = gammaNode->AddEntryVar(allocaResults[0]);
  auto memoryStateEntryVar = gammaNode->AddEntryVar(allocaResults[1]);
  auto ioStateEntryVar = gammaNode->AddEntryVar(ioStateImport);
  auto valueEntryVar = gammaNode->AddEntryVar(valueImport);

  auto & ioBarrierNode = jlm::rvsdg::CreateOpNode<IOBarrierOperation>(
      { addressEntryVar.branchArgument[0], ioStateEntryVar.branchArgument[0] },
      pointerType);

  auto & storeNode = StoreNonVolatileOperation::CreateNode(
      *ioBarrierNode.output(0),
      *valueEntryVar.branchArgument[0],
      { memoryStateEntryVar.branchArgument[0] },
      4);

  auto exitVar =
      gammaNode->AddExitVar({ storeNode.output(0), memoryStateEntryVar.branchArgument[1] });

  GraphExport::Create(*exitVar.output, "store");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto successStoreNode = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::NormalizeIOBarrierAllocaAddress,
      storeNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(successStoreNode);
  // There should only be the store node left.
  // The IOBarrier node should have been pruned.
  assert(gammaNode->subregion(0)->nnodes() == 1);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exitVar.branchResult[0]->origin())
          ->input(0)
          ->origin()
      == addressEntryVar.branchArgument[0]);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestIOBarrierAllocaAddressNormalization_Gamma",
    IOBarrierAllocaAddressNormalization_Gamma)
