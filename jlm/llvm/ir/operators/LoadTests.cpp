/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(LoadOperationTests, OperationEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();

  LoadNonVolatileOperation operation1(valueType, 2, 4);
  LoadNonVolatileOperation operation2(pointerType, 2, 4);
  LoadNonVolatileOperation operation3(valueType, 4, 4);
  LoadNonVolatileOperation operation4(valueType, 2, 8);
  TestOperation operation5({ PointerType::Create() }, { PointerType::Create() });

  // Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // loaded type differs
  EXPECT_NE(operation1, operation3); // number of memory states differs
  EXPECT_NE(operation1, operation4); // alignment differs
  EXPECT_NE(operation1, operation5); // operation differs
}

TEST(LoadOperationTests, TestCopy)
{
  using namespace jlm::llvm;

  // Arrange
  auto memoryType = MemoryStateType::Create();
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();

  jlm::rvsdg::Graph graph;
  auto address1 = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "address1");
  auto memoryState1 = &jlm::rvsdg::GraphImport::Create(graph, memoryType, "memoryState1");

  auto address2 = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "address2");
  auto memoryState2 = &jlm::rvsdg::GraphImport::Create(graph, memoryType, "memoryState2");

  auto loadResults = LoadNonVolatileOperation::Create(address1, { memoryState1 }, valueType, 4);

  // Act
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*loadResults[0]);
  EXPECT_TRUE(is<LoadNonVolatileOperation>(node));
  auto copiedNode = node->copy(&graph.GetRootRegion(), { address2, memoryState2 });

  // Assert
  EXPECT_EQ(
      node->GetOperation(),
      jlm::util::assertedCast<jlm::rvsdg::SimpleNode>(copiedNode)->GetOperation());
}

TEST(LoadOperationTests, TestLoadAllocaReduction)
{
  using namespace jlm::llvm;

  // Arrange
  auto mt = MemoryStateType::Create();
  auto bt = jlm::rvsdg::BitType::Create(32);

  jlm::rvsdg::Graph graph;
  auto size = &jlm::rvsdg::GraphImport::Create(graph, bt, "v");

  auto alloca1 = AllocaOperation::create(bt, size, 4);
  auto alloca2 = AllocaOperation::create(bt, size, 4);
  auto mux = MemoryStateMergeOperation::Create({ alloca1[1] });
  auto & loadNode =
      LoadNonVolatileOperation::CreateNode(*alloca1[0], { alloca1[1], alloca2[1], mux }, bt, 4);

  auto & ex = jlm::rvsdg::GraphExport::Create(*loadNode.output(0), "l");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadAlloca,
      loadNode);
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(node));
  EXPECT_EQ(node->ninputs(), 3u);
  EXPECT_EQ(node->input(1)->origin(), alloca1[1]);
  EXPECT_EQ(node->input(2)->origin(), mux);
}

TEST(LoadOperationTests, TestDuplicateStateReduction)
{
  using namespace jlm::llvm;

  // Arrange
  const auto memoryType = MemoryStateType::Create();
  const auto valueType = jlm::rvsdg::TestType::createValueType();
  const auto pointerType = PointerType::Create();

  jlm::rvsdg::Graph graph;
  const auto a = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "a");
  auto s1 = &jlm::rvsdg::GraphImport::Create(graph, memoryType, "s1");
  auto s2 = &jlm::rvsdg::GraphImport::Create(graph, memoryType, "s2");
  auto s3 = &jlm::rvsdg::GraphImport::Create(graph, memoryType, "s3");

  auto & loadNode = LoadNonVolatileOperation::CreateNode(*a, { s1, s2, s1, s2, s3 }, valueType, 4);

  auto & exA = jlm::rvsdg::GraphExport::Create(*loadNode.output(0), "exA");
  auto & exS1 = jlm::rvsdg::GraphExport::Create(*loadNode.output(1), "exS1");
  auto & exS2 = jlm::rvsdg::GraphExport::Create(*loadNode.output(2), "exS2");
  auto & exS3 = jlm::rvsdg::GraphExport::Create(*loadNode.output(3), "exS3");
  auto & exS4 = jlm::rvsdg::GraphExport::Create(*loadNode.output(4), "exS4");
  auto & exS5 = jlm::rvsdg::GraphExport::Create(*loadNode.output(5), "exS5");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeDuplicateStates,
      loadNode);

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_TRUE(success);
  const auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exA.origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(node));
  EXPECT_EQ(node->ninputs(), 4u);  // 1 address + 3 states
  EXPECT_EQ(node->noutputs(), 4u); // 1 loaded value + 3 states

  EXPECT_EQ(exA.origin(), node->output(0));
  EXPECT_EQ(exS1.origin(), node->output(1));
  EXPECT_EQ(exS2.origin(), node->output(2));
  EXPECT_EQ(exS3.origin(), node->output(1));
  EXPECT_EQ(exS4.origin(), node->output(2));
  EXPECT_EQ(exS5.origin(), node->output(3));
}

TEST(LoadOperationTests, TestLoadStoreStateReduction)
{
  using namespace jlm::llvm;

  // Arrange
  auto bt = jlm::rvsdg::BitType::Create(32);

  jlm::rvsdg::Graph graph;
  auto size = &jlm::rvsdg::GraphImport::Create(graph, bt, "v");

  auto alloca1 = AllocaOperation::create(bt, size, 4);
  auto alloca2 = AllocaOperation::create(bt, size, 4);
  auto store1 = StoreNonVolatileOperation::Create(alloca1[0], size, { alloca1[1] }, 4);
  auto store2 = StoreNonVolatileOperation::Create(alloca2[0], size, { alloca2[1] }, 4);

  auto & loadNode1 =
      LoadNonVolatileOperation::CreateNode(*alloca1[0], { store1[0], store2[0] }, bt, 4);
  auto & loadNode2 = LoadNonVolatileOperation::CreateNode(*alloca1[0], { store1[0] }, bt, 8);

  auto & ex1 = jlm::rvsdg::GraphExport::Create(*loadNode1.output(0), "l1");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*loadNode2.output(0), "l2");

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
  EXPECT_TRUE(success1);
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex1.origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(node));
  EXPECT_EQ(node->ninputs(), 2u);

  EXPECT_FALSE(success2);
  node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex2.origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(node));
  EXPECT_EQ(node->ninputs(), 2u);
}

TEST(LoadOperationTests, TestLoadStoreReduction_Success)
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::rvsdg::TestType::createValueType();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto a = &jlm::rvsdg::GraphImport::Create(graph, pt, "address");
  auto v = &jlm::rvsdg::GraphImport::Create(graph, vt, "value");
  auto s = &jlm::rvsdg::GraphImport::Create(graph, mt, "state");

  auto s1 = StoreNonVolatileOperation::Create(a, v, { s }, 4)[0];
  auto & loadNode = LoadNonVolatileOperation::CreateNode(*a, { s1 }, vt, 4);

  auto & x1 = jlm::rvsdg::GraphExport::Create(*loadNode.output(0), "value");
  auto & x2 = jlm::rvsdg::GraphExport::Create(*loadNode.output(1), "state");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadStore,
      loadNode);

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_TRUE(success);
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 1u);
  EXPECT_EQ(x1.origin(), v);
  EXPECT_EQ(x2.origin(), s1);
}

/**
 * Tests the load-store reduction with the value type of the store being different from the
 * value type of the load.
 */
TEST(LoadOperationTests, LoadStoreReduction_DifferentValueOperandType)
{
  using namespace jlm::llvm;

  // Arrange
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  jlm::rvsdg::Graph graph;
  auto & address = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address");
  auto & value = jlm::rvsdg::GraphImport::Create(graph, jlm::rvsdg::BitType::Create(32), "value");
  auto memoryState = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memoryState");

  auto & storeNode = StoreNonVolatileOperation::CreateNode(address, value, { memoryState }, 4);
  auto & loadNode = LoadNonVolatileOperation::CreateNode(
      address,
      outputs(&storeNode),
      jlm::rvsdg::BitType::Create(8),
      4);

  auto & exportedValue = jlm::rvsdg::GraphExport::Create(*loadNode.output(0), "v");
  jlm::rvsdg::GraphExport::Create(*loadNode.output(1), "s");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeLoadStore,
      loadNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_FALSE(success);

  const auto expectedLoadNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exportedValue.origin());
  EXPECT_EQ(expectedLoadNode, &loadNode);
  EXPECT_EQ(expectedLoadNode->ninputs(), 2u);

  const auto expectedStoreNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*expectedLoadNode->input(1)->origin());
  EXPECT_EQ(expectedStoreNode, &storeNode);
}

TEST(LoadOperationTests, IOBarrierAllocaAddressNormalization)
{
  using namespace jlm::llvm;

  // Arrange
  const auto valueType = jlm::rvsdg::TestType::createValueType();
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto bit32Type = jlm::rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();

  jlm::rvsdg::Graph graph;
  const auto addressImport = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "address");
  const auto sizeImport = &jlm::rvsdg::GraphImport::Create(graph, bit32Type, "value");
  auto memoryStateImport = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "memState");
  auto ioStateImport = &jlm::rvsdg::GraphImport::Create(graph, ioStateType, "ioState");

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

  auto & ex1 = jlm::rvsdg::GraphExport::Create(*loadNode1.output(0), "store1");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*loadNode2.output(0), "store2");

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
  EXPECT_TRUE(successLoadNode1);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex1.origin())->input(0)->origin(),
      allocaResults[0]);

  // There is no IOBarrierOperation node as producer for the load address. We expect the
  // normalization not to trigger.
  EXPECT_FALSE(successLoadNode2);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex2.origin())->input(0)->origin(),
      addressImport);
}

TEST(LoadOperationTests, IOBarrierAllocaAddressNormalization_Gamma)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto bit32Type = jlm::rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto controlType = ControlType::Create(2);

  Graph graph;
  const auto sizeImport = &jlm::rvsdg::GraphImport::Create(graph, bit32Type, "value");
  const auto controlImport = &jlm::rvsdg::GraphImport::Create(graph, controlType, "control");
  auto ioStateImport = &jlm::rvsdg::GraphImport::Create(graph, ioStateType, "ioState");
  auto valueImport = &jlm::rvsdg::GraphImport::Create(graph, valueType, "value");

  auto allocaResults = AllocaOperation::create(valueType, sizeImport, 4);

  auto gammaNode = GammaNode::create(controlImport, 2);
  auto addressEntryVar = gammaNode->AddEntryVar(allocaResults[0]);
  auto memoryStateEntryVar = gammaNode->AddEntryVar(allocaResults[1]);
  auto ioStateEntryVar = gammaNode->AddEntryVar(ioStateImport);
  auto valueEntryVar = gammaNode->AddEntryVar(valueImport);

  auto & ioBarrierNode = jlm::rvsdg::CreateOpNode<IOBarrierOperation>(
      { addressEntryVar.branchArgument[0], ioStateEntryVar.branchArgument[0] },
      pointerType);

  auto & loadNode = LoadNonVolatileOperation::CreateNode(
      *ioBarrierNode.output(0),
      { memoryStateEntryVar.branchArgument[0] },
      valueType,
      4);

  auto exitVar = gammaNode->AddExitVar({ loadNode.output(0), valueEntryVar.branchArgument[1] });

  GraphExport::Create(*exitVar.output, "load1");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto successLoadNode = jlm::rvsdg::ReduceNode<LoadNonVolatileOperation>(
      LoadNonVolatileOperation::NormalizeIOBarrierAllocaAddress,
      loadNode);

  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_TRUE(successLoadNode);
  // There should only be the load node left.
  // The IOBarrier node should have been pruned.
  EXPECT_EQ(gammaNode->subregion(0)->numNodes(), 1u);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exitVar.branchResult[0]->origin())
          ->input(0)
          ->origin(),
      addressEntryVar.branchArgument[0]);
}

TEST(LoadOperationTests, LoadVolatileOperationEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();

  LoadVolatileOperation operation1(valueType, 2, 4);
  LoadVolatileOperation operation2(pointerType, 2, 4);
  LoadVolatileOperation operation3(valueType, 4, 4);
  LoadVolatileOperation operation4(valueType, 2, 8);
  TestOperation operation5({ PointerType::Create() }, { PointerType::Create() });

  // Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // loaded type differs
  EXPECT_NE(operation1, operation3); // number of memory states differs
  EXPECT_NE(operation1, operation4); // alignment differs
  EXPECT_NE(operation1, operation5); // operation differs
}

TEST(LoadOperationTests, OperationCopy)
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::rvsdg::TestType::createValueType();
  PointerType pointerType;

  LoadVolatileOperation operation(valueType, 2, 4);

  // Act
  auto copiedOperation = operation.copy();

  // Assert
  EXPECT_EQ(*copiedOperation, operation);
}

TEST(LoadOperationTests, OperationAccessors)
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::rvsdg::TestType::createValueType();
  PointerType pointerType;

  size_t alignment = 4;
  size_t numMemoryStates = 2;
  LoadVolatileOperation operation(valueType, numMemoryStates, alignment);

  // Assert
  EXPECT_EQ(operation.GetLoadedType(), valueType);
  EXPECT_EQ(operation.NumMemoryStates(), numMemoryStates);
  EXPECT_EQ(operation.GetAlignment(), alignment);
  EXPECT_EQ(operation.narguments(), numMemoryStates + 2); // [address, ioState, memoryStates]
  EXPECT_EQ(operation.nresults(), numMemoryStates + 2);   // [loadedValue, ioState, memoryStates]
}

TEST(LoadOperationTests, NodeCopy)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto pointerType = PointerType::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryType = MemoryStateType::Create();
  auto valueType = TestType::createValueType();

  jlm::rvsdg::Graph graph;
  auto & address1 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address1");
  auto & iOState1 = jlm::rvsdg::GraphImport::Create(graph, iOStateType, "iOState1");
  auto & memoryState1 = jlm::rvsdg::GraphImport::Create(graph, memoryType, "memoryState1");

  auto & address2 = jlm::rvsdg::GraphImport::Create(graph, pointerType, "address2");
  auto & iOState2 = jlm::rvsdg::GraphImport::Create(graph, iOStateType, "iOState2");
  auto & memoryState2 = jlm::rvsdg::GraphImport::Create(graph, memoryType, "memoryState2");

  auto & loadNode = jlm::rvsdg::CreateOpNode<LoadVolatileOperation>(
      { &address1, &iOState1, &memoryState1 },
      valueType,
      1,
      4);

  // Act
  auto copiedNode = loadNode.copy(&graph.GetRootRegion(), { &address2, &iOState2, &memoryState2 });

  // Assert
  auto copiedOperation = dynamic_cast<const LoadVolatileOperation *>(
      &jlm::util::assertedCast<SimpleNode>(copiedNode)->GetOperation());
  EXPECT_NE(copiedOperation, nullptr);
  EXPECT_EQ(LoadOperation::AddressInput(*copiedNode).origin(), &address2);
  EXPECT_EQ(LoadVolatileOperation::IOStateInput(*copiedNode).origin(), &iOState2);
  EXPECT_EQ(*copiedOperation->GetLoadedType(), *valueType);
}
