/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/MemCpy.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(MemCpyOperationTests, operationEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto bit32Type = jlm::rvsdg::BitType::Create(32);
  auto bit64Type = jlm::rvsdg::BitType::Create(64);

  MemCpyNonVolatileOperation operation1(bit32Type, 1);
  MemCpyNonVolatileOperation operation2(bit64Type, 4);
  TestOperation operation3({ valueType }, { valueType });

  // Act & Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // length type differs
  EXPECT_NE(operation1, operation3); // number of memory states differs
}

TEST(MemCpyOperationTests, accessors)
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
  EXPECT_EQ(MemCpyOperation::destinationInput(memCpy).origin(), &address1);
  EXPECT_EQ(MemCpyOperation::sourceInput(memCpy).origin(), &address2);
  EXPECT_EQ(MemCpyOperation::countInput(memCpy).origin(), constant100.output(0));
}

TEST(MemCpyOperationTests, mapMemoryStateInputToOutput)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto bit32Type = jlm::rvsdg::BitType::Create(32);
  auto ioStateType = IOStateType::Create();
  auto memStateType = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto & address1 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address1");
  auto & address2 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address2");
  auto & ioState = jlm::rvsdg::GraphImport::Create(graph, ioStateType, "ioState");
  auto & memState1 = jlm::rvsdg::GraphImport::Create(graph, memStateType, "memState1");
  auto & memState2 = jlm::rvsdg::GraphImport::Create(graph, memStateType, "memState2");

  auto & constant100 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100);

  auto & memCpyNonVolatile = MemCpyNonVolatileOperation::createNode(
      address1,
      address2,
      *constant100.output(0),
      { &memState1, &memState2 });

  auto & memCpyVolatile = MemCpyVolatileOperation::CreateNode(
      address1,
      address2,
      *constant100.output(0),
      ioState,
      { &memState1, &memState2 });

  // Act & Assert
  EXPECT_EQ(
      &MemCpyOperation::mapMemoryStateInputToOutput(*memCpyNonVolatile.input(3)),
      memCpyNonVolatile.output(0));
  EXPECT_EQ(
      &MemCpyOperation::mapMemoryStateInputToOutput(*memCpyNonVolatile.input(4)),
      memCpyNonVolatile.output(1));

  EXPECT_EQ(
      &MemCpyOperation::mapMemoryStateInputToOutput(*memCpyVolatile.input(4)),
      memCpyVolatile.output(1));
  EXPECT_EQ(
      &MemCpyOperation::mapMemoryStateInputToOutput(*memCpyVolatile.input(5)),
      memCpyVolatile.output(2));
}

TEST(MemCpyOperationTests, mapMemoryStateOutputToInput)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto bit32Type = jlm::rvsdg::BitType::Create(32);
  auto ioStateType = IOStateType::Create();
  auto memStateType = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto & address1 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address1");
  auto & address2 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address2");
  auto & ioState = jlm::rvsdg::GraphImport::Create(graph, ioStateType, "ioState");
  auto & memState1 = jlm::rvsdg::GraphImport::Create(graph, memStateType, "memState1");
  auto & memState2 = jlm::rvsdg::GraphImport::Create(graph, memStateType, "memState2");

  auto & constant100 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100);

  auto & memCpyNonVolatile = MemCpyNonVolatileOperation::createNode(
      address1,
      address2,
      *constant100.output(0),
      { &memState1, &memState2 });

  auto & memCpyVolatile = MemCpyVolatileOperation::CreateNode(
      address1,
      address2,
      *constant100.output(0),
      ioState,
      { &memState1, &memState2 });

  // Act & Assert
  EXPECT_EQ(
      MemCpyOperation::mapMemoryStateOutputToInput(*memCpyNonVolatile.output(0)).origin(),
      &memState1);
  EXPECT_EQ(
      MemCpyOperation::mapMemoryStateOutputToInput(*memCpyNonVolatile.output(1)).origin(),
      &memState2);

  EXPECT_EQ(
      MemCpyOperation::mapMemoryStateOutputToInput(*memCpyVolatile.output(1)).origin(),
      &memState1);
  EXPECT_EQ(
      MemCpyOperation::mapMemoryStateOutputToInput(*memCpyVolatile.output(2)).origin(),
      &memState2);
}
