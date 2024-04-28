/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/LoadVolatile.hpp>

static int
OperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;
  PointerType pointerType;

  LoadVolatileOperation operation1(valueType, 2, 4);
  LoadVolatileOperation operation2(pointerType, 2, 4);
  LoadVolatileOperation operation3(valueType, 4, 4);
  LoadVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::test_op operation5({ &pointerType }, { &pointerType });

  // Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // loaded type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadVolatileTests-OperationEquality",
    OperationEquality)

static int
OperationCopy()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;
  PointerType pointerType;

  LoadVolatileOperation operation(valueType, 2, 4);

  // Act
  auto copiedOperation = operation.copy();

  // Assert
  assert(*copiedOperation == operation);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/LoadVolatileTests-OperationCopy", OperationCopy)

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
  LoadVolatileOperation operation(valueType, numMemoryStates, alignment);

  // Assert
  assert(operation.GetLoadedType() == valueType);
  assert(operation.NumMemoryStates() == numMemoryStates);
  assert(operation.GetAlignment() == alignment);
  assert(operation.narguments() == numMemoryStates + 2); // [address, ioState, memoryStates]
  assert(operation.nresults() == numMemoryStates + 2);   // [loadedValue, ioState, memoryStates]

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/LoadVolatileTests-OperationAccessors",
    OperationAccessors)

static int
NodeCopy()
{
  using namespace jlm::llvm;

  // Arrange
  PointerType pointerType;
  iostatetype iOStateType;
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;

  jlm::rvsdg::graph graph;
  auto & address1 = *graph.add_import({ pointerType, "address1" });
  auto & iOState1 = *graph.add_import({ iOStateType, "iOState1" });
  auto & memoryState1 = *graph.add_import({ memoryType, "memoryState1" });

  auto & address2 = *graph.add_import({ pointerType, "address2" });
  auto & iOState2 = *graph.add_import({ iOStateType, "iOState2" });
  auto & memoryState2 = *graph.add_import({ memoryType, "memoryState2" });

  auto & loadNode =
      LoadVolatileNode::CreateNode(address1, iOState1, { &memoryState1 }, valueType, 4);

  // Act
  auto copiedNode = loadNode.copy(graph.root(), { &address2, &iOState2, &memoryState2 });

  // Assert
  auto copiedLoadNode = dynamic_cast<const LoadVolatileNode *>(copiedNode);
  assert(loadNode.GetOperation() == copiedLoadNode->GetOperation());
  assert(copiedLoadNode->GetAddressInput().origin() == &address2);
  assert(copiedLoadNode->GetIoStateInput().origin() == &iOState2);
  assert(copiedLoadNode->GetLoadedValueOutput().type() == valueType);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/LoadVolatileTests-NodeCopy", NodeCopy)
