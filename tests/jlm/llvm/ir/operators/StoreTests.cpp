/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

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
  auto ioStateType = iostatetype::Create();
  auto memoryType = MemoryStateType::Create();
  auto valueType = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  auto & address1 = jlm::tests::GraphImport::Create(graph, pointerType, "address1");
  auto & value1 = jlm::tests::GraphImport::Create(graph, valueType, "value1");
  auto & ioState1 = jlm::tests::GraphImport::Create(graph, ioStateType, "ioState1");
  auto & memoryState1 = jlm::tests::GraphImport::Create(graph, memoryType, "memoryState1");

  auto & address2 = jlm::tests::GraphImport::Create(graph, pointerType, "address2");
  auto & value2 = jlm::tests::GraphImport::Create(graph, valueType, "value2");
  auto & ioState2 = jlm::tests::GraphImport::Create(graph, ioStateType, "ioState2");
  auto & memoryState2 = jlm::tests::GraphImport::Create(graph, memoryType, "memoryState2");

  auto & storeNode =
      StoreVolatileNode::CreateNode(address1, value1, ioState1, { &memoryState1 }, 4);

  // Act
  auto copiedNode = storeNode.copy(graph.root(), { &address2, &value2, &ioState2, &memoryState2 });

  // Assert
  auto copiedStoreNode = dynamic_cast<const StoreVolatileNode *>(copiedNode);
  assert(storeNode.GetOperation() == copiedStoreNode->GetOperation());
  assert(copiedStoreNode->GetAddressInput().origin() == &address2);
  assert(copiedStoreNode->GetStoredValueInput().origin() == &value2);
  assert(copiedStoreNode->GetIoStateInput().origin() == &ioState2);
  assert(copiedStoreNode->GetIoStateOutput().type() == *ioStateType);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreTests-StoreVolatileNodeCopy",
    StoreVolatileNodeCopy)

static void
TestCopy()
{
  using namespace jlm::llvm;

  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::rvsdg::graph graph;
  auto address1 = &jlm::tests::GraphImport::Create(graph, pointerType, "address1");
  auto value1 = &jlm::tests::GraphImport::Create(graph, valueType, "value1");
  auto memoryState1 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "state1");

  auto address2 = &jlm::tests::GraphImport::Create(graph, pointerType, "address2");
  auto value2 = &jlm::tests::GraphImport::Create(graph, valueType, "value2");
  auto memoryState2 = &jlm::tests::GraphImport::Create(graph, memoryStateType, "state2");

  auto storeResults = StoreNonVolatileNode::Create(address1, value1, { memoryState1 }, 4);

  // Act
  auto node = jlm::rvsdg::node_output::node(storeResults[0]);
  auto storeNode = jlm::util::AssertedCast<const StoreNonVolatileNode>(node);
  auto copiedNode = storeNode->copy(graph.root(), { address2, value2, memoryState2 });

  // Assert
  auto copiedStoreNode = dynamic_cast<const StoreNonVolatileNode *>(copiedNode);
  assert(copiedNode != nullptr);
  assert(storeNode->GetOperation() == copiedStoreNode->GetOperation());
}

static void
TestStoreMuxReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::graph graph;
  auto nf = graph.node_normal_form(typeid(StoreNonVolatileOperation));
  auto snf = static_cast<jlm::llvm::store_normal_form *>(nf);
  snf->set_mutable(false);
  snf->set_store_mux_reducible(false);

  auto a = &jlm::tests::GraphImport::Create(graph, pt, "a");
  auto v = &jlm::tests::GraphImport::Create(graph, vt, "v");
  auto s1 = &jlm::tests::GraphImport::Create(graph, mt, "s1");
  auto s2 = &jlm::tests::GraphImport::Create(graph, mt, "s2");
  auto s3 = &jlm::tests::GraphImport::Create(graph, mt, "s3");

  auto mux = MemoryStateMergeOperation::Create({ s1, s2, s3 });
  auto state = StoreNonVolatileNode::Create(a, v, { mux }, 4);

  auto & ex = GraphExport::Create(*state[0], "s");

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  snf->set_mutable(true);
  snf->set_store_mux_reducible(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto muxnode = jlm::rvsdg::node_output::node(ex.origin());
  assert(is<MemoryStateMergeOperation>(muxnode));
  assert(muxnode->ninputs() == 3);
  auto n0 = jlm::rvsdg::node_output::node(muxnode->input(0)->origin());
  auto n1 = jlm::rvsdg::node_output::node(muxnode->input(1)->origin());
  auto n2 = jlm::rvsdg::node_output::node(muxnode->input(2)->origin());
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n0->operation()));
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n1->operation()));
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(n2->operation()));
}

static void
TestMultipleOriginReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::graph graph;
  auto nf = graph.node_normal_form(typeid(StoreNonVolatileOperation));
  auto snf = static_cast<jlm::llvm::store_normal_form *>(nf);
  snf->set_mutable(false);
  snf->set_multiple_origin_reducible(false);

  auto a = &jlm::tests::GraphImport::Create(graph, pt, "a");
  auto v = &jlm::tests::GraphImport::Create(graph, vt, "v");
  auto s = &jlm::tests::GraphImport::Create(graph, mt, "s");

  auto states = StoreNonVolatileNode::Create(a, v, { s, s, s, s }, 4);

  auto & ex = GraphExport::Create(*states[0], "s");

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  snf->set_mutable(true);
  snf->set_multiple_origin_reducible(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto node = jlm::rvsdg::node_output::node(ex.origin());
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(node->operation()) && node->ninputs() == 3);
}

static void
TestStoreAllocaReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto mt = MemoryStateType::Create();
  auto bt = jlm::rvsdg::bittype::Create(32);

  jlm::rvsdg::graph graph;
  auto nf = graph.node_normal_form(typeid(StoreNonVolatileOperation));
  auto snf = static_cast<jlm::llvm::store_normal_form *>(nf);
  snf->set_mutable(false);
  snf->set_store_alloca_reducible(false);

  auto size = &jlm::tests::GraphImport::Create(graph, bt, "size");
  auto value = &jlm::tests::GraphImport::Create(graph, vt, "value");
  auto s = &jlm::tests::GraphImport::Create(graph, mt, "s");

  auto alloca1 = alloca_op::create(vt, size, 4);
  auto alloca2 = alloca_op::create(vt, size, 4);
  auto states1 = StoreNonVolatileNode::Create(alloca1[0], value, { alloca1[1], alloca2[1], s }, 4);
  auto states2 = StoreNonVolatileNode::Create(alloca2[0], value, states1, 4);

  GraphExport::Create(*states2[0], "s1");
  GraphExport::Create(*states2[1], "s2");
  GraphExport::Create(*states2[2], "s3");

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  snf->set_mutable(true);
  snf->set_store_alloca_reducible(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  bool has_add_import = false;
  for (size_t n = 0; n < graph.root()->nresults(); n++)
  {
    if (graph.root()->result(n)->origin() == s)
      has_add_import = true;
  }
  assert(has_add_import);
}

static void
TestStoreStoreReduction()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::graph graph;
  auto a = &jlm::tests::GraphImport::Create(graph, pt, "address");
  auto v1 = &jlm::tests::GraphImport::Create(graph, vt, "value");
  auto v2 = &jlm::tests::GraphImport::Create(graph, vt, "value");
  auto s = &jlm::tests::GraphImport::Create(graph, mt, "state");

  auto s1 = StoreNonVolatileNode::Create(a, v1, { s }, 4)[0];
  auto s2 = StoreNonVolatileNode::Create(a, v2, { s1 }, 4)[0];

  auto & ex = GraphExport::Create(*s2, "state");

  jlm::rvsdg::view(graph.root(), stdout);

  // Act
  auto nf = StoreNonVolatileOperation::GetNormalForm(&graph);
  nf->set_store_store_reducible(true);
  graph.normalize();
  graph.prune();

  jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  assert(graph.root()->nnodes() == 1);
  assert(jlm::rvsdg::node_output::node(ex.origin())->input(1)->origin() == v2);
}

static int
TestStore()
{
  TestCopy();

  TestStoreMuxReduction();
  TestStoreAllocaReduction();
  TestMultipleOriginReduction();
  TestStoreStoreReduction();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/StoreTests", TestStore)
