/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

static int
StoreNonVolatileOperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();

  StoreNonVolatileOperation operation1(valueType, 2, 4);
  StoreNonVolatileOperation operation2(pointerType, 2, 4);
  StoreNonVolatileOperation operation3(valueType, 4, 4);
  StoreNonVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::test_op operation5({ PointerType::Create() }, { PointerType::Create() });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // stored type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreNonVolatileOperationEquality",
    StoreNonVolatileOperationEquality)

static int
StoreVolatileOperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();

  StoreVolatileOperation operation1(valueType, 2, 4);
  StoreVolatileOperation operation2(pointerType, 2, 4);
  StoreVolatileOperation operation3(valueType, 4, 4);
  StoreVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::test_op operation5({ PointerType::Create() }, { PointerType::Create() });

  // Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // stored type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileOperationEquality",
    StoreVolatileOperationEquality)

static int
StoreVolatileOperationCopy()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::valuetype::Create();
  PointerType pointerType;

  StoreVolatileOperation operation(valueType, 2, 4);

  // Act
  auto copiedOperation = operation.copy();

  // Assert
  assert(*copiedOperation == operation);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileOperationCopy",
    StoreVolatileOperationCopy)

static int
StoreVolatileOperationAccessors()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::tests::valuetype::Create();
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

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileOperationAccessors",
    StoreVolatileOperationAccessors)

static int
StoreVolatileNodeCopy()
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto memoryType = MemoryStateType::Create();
  auto valueType = jlm::tests::valuetype::Create();

  jlm::rvsdg::Graph graph;
  auto & address1 = jlm::tests::GraphImport::Create(graph, pointerType, "address1");
  auto & value1 = jlm::tests::GraphImport::Create(graph, valueType, "value1");
  auto & ioState1 = jlm::tests::GraphImport::Create(graph, ioStateType, "ioState1");
  auto & memoryState1 = jlm::tests::GraphImport::Create(graph, memoryType, "memoryState1");

  auto & address2 = jlm::tests::GraphImport::Create(graph, pointerType, "address2");
  auto & value2 = jlm::tests::GraphImport::Create(graph, valueType, "value2");
  auto & ioState2 = jlm::tests::GraphImport::Create(graph, ioStateType, "ioState2");
  auto & memoryState2 = jlm::tests::GraphImport::Create(graph, memoryType, "memoryState2");

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
  assert(StoreVolatileOperation::IOStateOutput(*copiedNode).type() == *ioStateType);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileNodeCopy",
    StoreVolatileNodeCopy)

static int
TestCopy()
{
  using namespace jlm::llvm;

  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto address1 = &jlm::tests::GraphImport::Create(graph, pointerType, "address1");
  auto value1 = &jlm::tests::GraphImport::Create(graph, valueType, "value1");
  auto memoryState1 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "state1");

  auto address2 = &jlm::tests::GraphImport::Create(graph, pointerType, "address2");
  auto value2 = &jlm::tests::GraphImport::Create(graph, valueType, "value2");
  auto memoryState2 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "state2");

  auto storeResults = StoreNonVolatileOperation::Create(address1, value1, { memoryState1 }, 4);

  // Act
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeResults[0]);
  auto copiedNode = node->copy(&graph.GetRootRegion(), { address2, value2, memoryState2 });

  // Assert
  assert(node->GetOperation() == copiedNode->GetOperation());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/StoreTests-TestCopy", TestCopy)

static int
TestStoreMuxNormalization()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a = &jlm::tests::GraphImport::Create(graph, pt, "a");
  auto v = &jlm::tests::GraphImport::Create(graph, vt, "v");
  auto s1 = &jlm::tests::GraphImport::Create(graph, mt, "s1");
  auto s2 = &jlm::tests::GraphImport::Create(graph, mt, "s2");
  auto s3 = &jlm::tests::GraphImport::Create(graph, mt, "s3");

  auto mux = MemoryStateMergeOperation::Create({ s1, s2, s3 });
  auto & storeNode = StoreNonVolatileOperation::CreateNode(*a, *v, { mux }, 4);

  auto & ex = GraphExport::Create(*storeNode.output(0), "s");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  auto success = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(NormalizeStoreMux, storeNode);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  auto muxNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin());
  assert(is<MemoryStateMergeOperation>(muxNode));
  assert(muxNode->ninputs() == 3);
  auto n0 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*muxNode->input(0)->origin());
  auto n1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*muxNode->input(1)->origin());
  auto n2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*muxNode->input(2)->origin());
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n0->GetOperation()));
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n1->GetOperation()));
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n2->GetOperation()));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestStoreMuxNormalization",
    TestStoreMuxNormalization)

static int
TestDuplicateStateReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a = &jlm::tests::GraphImport::Create(graph, pointerType, "a");
  auto v = &jlm::tests::GraphImport::Create(graph, valueType, "v");
  auto s1 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "s1");
  auto s2 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "s2");
  auto s3 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "s3");

  auto & storeNode = StoreNonVolatileOperation::CreateNode(*a, *v, { s1, s2, s1, s2, s3 }, 4);

  auto & exS1 = GraphExport::Create(*storeNode.output(0), "exS1");
  auto & exS2 = GraphExport::Create(*storeNode.output(1), "exS2");
  auto & exS3 = GraphExport::Create(*storeNode.output(2), "exS3");
  auto & exS4 = GraphExport::Create(*storeNode.output(3), "exS4");
  auto & exS5 = GraphExport::Create(*storeNode.output(4), "exS5");

  view(&graph.GetRootRegion(), stdout);

  // Act
  auto success =
      jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(NormalizeStoreDuplicateState, storeNode);
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

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestDuplicateStateReduction",
    TestDuplicateStateReduction)

static int
TestStoreAllocaReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto mt = MemoryStateType::Create();
  auto bt = jlm::rvsdg::bittype::Create(32);

  jlm::rvsdg::Graph graph;
  auto size = &jlm::tests::GraphImport::Create(graph, bt, "size");
  auto value = &jlm::tests::GraphImport::Create(graph, vt, "value");
  auto s = &jlm::tests::GraphImport::Create(graph, mt, "s");

  auto alloca1 = alloca_op::create(vt, size, 4);
  auto alloca2 = alloca_op::create(vt, size, 4);
  auto & storeNode1 =
      StoreNonVolatileOperation::CreateNode(*alloca1[0], *value, { alloca1[1], alloca2[1], s }, 4);
  auto & storeNode2 =
      StoreNonVolatileOperation::CreateNode(*alloca2[0], *value, outputs(&storeNode1), 4);

  GraphExport::Create(*storeNode2.output(0), "s1");
  GraphExport::Create(*storeNode2.output(1), "s2");
  GraphExport::Create(*storeNode2.output(2), "s3");

  view(&graph.GetRootRegion(), stdout);

  // Act
  auto success1 =
      jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(NormalizeStoreAlloca, storeNode1);
  auto success2 =
      jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(NormalizeStoreAlloca, storeNode2);
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

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestStoreAllocaReduction",
    TestStoreAllocaReduction)

static int
TestStoreStoreReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a = &jlm::tests::GraphImport::Create(graph, pt, "address");
  auto v1 = &jlm::tests::GraphImport::Create(graph, vt, "value");
  auto v2 = &jlm::tests::GraphImport::Create(graph, vt, "value");
  auto s = &jlm::tests::GraphImport::Create(graph, mt, "state");

  auto & storeNode1 = StoreNonVolatileOperation::CreateNode(*a, *v1, { s }, 4);
  auto & storeNode2 = StoreNonVolatileOperation::CreateNode(*a, *v2, outputs(&storeNode1), 4);

  auto & ex = GraphExport::Create(*storeNode2.output(0), "state");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  auto success = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(NormalizeStoreStore, storeNode2);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  assert(graph.GetRootRegion().nnodes() == 1);
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin())->input(1)->origin() == v2);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-TestStoreStoreReduction",
    TestStoreStoreReduction)
