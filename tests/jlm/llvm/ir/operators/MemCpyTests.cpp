/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/MemCpy.hpp>

static void
operationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto bit32Type = jlm::rvsdg::BitType::Create(32);
  auto bit64Type = jlm::rvsdg::BitType::Create(64);

  MemCpyNonVolatileOperation operation1(bit32Type, 1);
  MemCpyNonVolatileOperation operation2(bit64Type, 4);
  jlm::tests::TestOperation operation3({ valueType }, { valueType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // length type differs
  assert(operation1 != operation3); // number of memory states differs
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/MemCpyNonVolatileTests-operationEquality",
    operationEquality)

static void
accessors()
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto bit32Type = jlm::rvsdg::BitType::Create(32);
  auto memStateType = jlm::llvm::MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto & address1 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address1");
  auto & address2 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address2");
  auto & memState = jlm::rvsdg::GraphImport::Create(graph, memStateType, "memState");
  auto & constant100 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100);

  auto & memCpy = jlm::rvsdg::CreateOpNode<MemCpyNonVolatileOperation>(
      { &address1, &address2, constant100.output(0), &memState },
      bit32Type,
      1);

  // Act & Assert
  assert(MemCpyOperation::destinationInput(memCpy).origin() == &address1);
  assert(MemCpyOperation::sourceInput(memCpy).origin() == &address2);
  assert(MemCpyOperation::countInput(memCpy).origin() == constant100.output(0));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/MemCpyNonVolatileTests-accessors", accessors)
