/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/StoreVolatile.hpp>

static int
OperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;
  PointerType pointerType;

  StoreVolatileOperation operation1(valueType, 2, 4);
  StoreVolatileOperation operation2(pointerType, 2, 4);
  StoreVolatileOperation operation3(valueType, 4, 4);
  StoreVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::test_op operation5({ &pointerType }, { &pointerType });

  // Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // stored type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreVolatileTests-OperationEquality",
    OperationEquality)

static int
OperationCopy()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;
  PointerType pointerType;

  StoreVolatileOperation operation(valueType, 2, 4);

  // Act
  auto copiedOperation = operation.copy();

  // Assert
  assert(*copiedOperation == operation);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/StoreVolatileTests-OperationCopy", OperationCopy)

static int
OperationAccessors()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;
  PointerType pointerType;

  size_t alignment = 4;
  size_t numMemoryStates = 2;
  StoreVolatileOperation operation(valueType, numMemoryStates, alignment);

  // Assert
  assert(operation.GetStoredType() == valueType);
  assert(operation.NumMemoryStates() == numMemoryStates);
  assert(operation.GetAlignment() == alignment);
  assert(
      operation.narguments()
      == numMemoryStates + 3); // [address, storedValue, ioState, memoryStates]
  assert(operation.nresults() == numMemoryStates + 1); // [ioState, memoryStates]

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreVolatileTests-OperationAccessors",
    OperationAccessors)

static int
NodeCopy()
{
  using namespace jlm::llvm;

  // Arrange
  PointerType pointerType;
  iostatetype ioStateType;
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;

  jlm::rvsdg::graph graph;
  auto & address1 = *graph.add_import({ pointerType, "address1" });
  auto & value1 = *graph.add_import({ valueType, "value1" });
  auto & ioState1 = *graph.add_import({ ioStateType, "ioState1" });
  auto & memoryState1 = *graph.add_import({ memoryType, "memoryState1" });

  auto & address2 = *graph.add_import({ pointerType, "address2" });
  auto & value2 = *graph.add_import({ valueType, "value2" });
  auto & ioState2 = *graph.add_import({ ioStateType, "ioState2" });
  auto & memoryState2 = *graph.add_import({ memoryType, "memoryState2" });

  auto & storeNode =
      StoreVolatileNode::CreateNode(address1, value1, ioState1, { &memoryState1 }, 4);

  // Act
  auto copiedNode = storeNode.copy(graph.root(), { &address2, &value2, &ioState2, &memoryState2 });

  // Assert
  auto copiedStoreNode = dynamic_cast<const StoreVolatileNode *>(copiedNode);
  assert(storeNode.GetOperation() == copiedStoreNode->GetOperation());
  assert(copiedStoreNode->GetAddressInput().origin() == &address2);
  assert(copiedStoreNode->GetValueInput().origin() == &value2);
  assert(copiedStoreNode->GetIoStateInput().origin() == &ioState2);
  assert(copiedStoreNode->GetIoStateOutput().type() == ioStateType);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/StoreVolatileTests-NodeCopy", NodeCopy)
