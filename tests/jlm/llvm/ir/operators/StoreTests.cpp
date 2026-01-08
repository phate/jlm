/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(StoreOperationTests, StoreNonVolatileOperationEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();

  StoreNonVolatileOperation operation1(valueType, 2, 4);
  StoreNonVolatileOperation operation2(pointerType, 2, 4);
  StoreNonVolatileOperation operation3(valueType, 4, 4);
  StoreNonVolatileOperation operation4(valueType, 2, 8);
  TestOperation operation5({ PointerType::Create() }, { PointerType::Create() });

  // Act & Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // stored type differs
  EXPECT_NE(operation1, operation3); // number of memory states differs
  EXPECT_NE(operation1, operation4); // alignment differs
  EXPECT_NE(operation1, operation5); // operation differs
}

TEST(StoreOperationTests, StoreVolatileOperationEquality)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();

  StoreVolatileOperation operation1(valueType, 2, 4);
  StoreVolatileOperation operation2(pointerType, 2, 4);
  StoreVolatileOperation operation3(valueType, 4, 4);
  StoreVolatileOperation operation4(valueType, 2, 8);
  TestOperation operation5({ PointerType::Create() }, { PointerType::Create() });

  // Assert
  EXPECT_EQ(operation1, operation1);
  EXPECT_NE(operation1, operation2); // stored type differs
  EXPECT_NE(operation1, operation3); // number of memory states differs
  EXPECT_NE(operation1, operation4); // alignment differs
  EXPECT_NE(operation1, operation5); // operation differs
}

TEST(StoreOperationTests, StoreVolatileOperationCopy)
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::rvsdg::TestType::createValueType();
  PointerType pointerType;

  StoreVolatileOperation operation(valueType, 2, 4);

  // Act
  auto copiedOperation = operation.copy();

  // Assert
  EXPECT_EQ(*copiedOperation, operation);
}

TEST(StoreOperationTests, StoreVolatileOperationAccessors)
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  auto valueType = jlm::rvsdg::TestType::createValueType();
  PointerType pointerType;

  size_t alignment = 4;
  size_t numMemoryStates = 2;
  StoreVolatileOperation operation(valueType, numMemoryStates, alignment);

  // Assert
  EXPECT_EQ(operation.GetStoredType(), *valueType);
  EXPECT_EQ(operation.NumMemoryStates(), numMemoryStates);
  EXPECT_EQ(operation.GetAlignment(), alignment);
  EXPECT_EQ(
      operation.narguments(),
      numMemoryStates + 3); // [address, storedValue, ioState, memoryStates]
  EXPECT_EQ(operation.nresults(), numMemoryStates + 1); // [ioState, memoryStates]
}

TEST(StoreOperationTests, StoreVolatileNodeCopy)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto memoryType = MemoryStateType::Create();
  auto valueType = jlm::rvsdg::TestType::createValueType();

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
  EXPECT_EQ(storeNode.GetOperation(), storeNode.GetOperation());
  EXPECT_EQ(StoreOperation::AddressInput(*copiedNode).origin(), &address2);
  EXPECT_EQ(StoreOperation::StoredValueInput(*copiedNode).origin(), &value2);
  EXPECT_EQ(StoreVolatileOperation::IOStateInput(*copiedNode).origin(), &ioState2);
  EXPECT_EQ(*StoreVolatileOperation::IOStateOutput(*copiedNode).Type(), *ioStateType);
}

TEST(StoreOperationTests, TestCopy)
{
  using namespace jlm::llvm;

  auto valueType = jlm::rvsdg::TestType::createValueType();
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
  EXPECT_EQ(
      node->GetOperation(),
      jlm::util::assertedCast<jlm::rvsdg::SimpleNode>(copiedNode)->GetOperation());
}

TEST(StoreOperationTests, TestStoreMuxNormalization)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = TestType::createValueType();
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
  EXPECT_TRUE(success);
  auto muxNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin());
  EXPECT_TRUE(is<MemoryStateMergeOperation>(muxNode));
  EXPECT_EQ(muxNode->ninputs(), 3u);
  auto n0 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*muxNode->input(0)->origin());
  auto n1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*muxNode->input(1)->origin());
  auto n2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*muxNode->input(2)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<StoreNonVolatileOperation>(n0->GetOperation()));
  EXPECT_TRUE(jlm::rvsdg::is<StoreNonVolatileOperation>(n1->GetOperation()));
  EXPECT_TRUE(jlm::rvsdg::is<StoreNonVolatileOperation>(n2->GetOperation()));
}

TEST(StoreOperationTests, TestDuplicateStateReduction)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();
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
  EXPECT_TRUE(success);
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exS1.origin());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(node));
  EXPECT_EQ(node->ninputs(), 5u);
  EXPECT_EQ(node->noutputs(), 3u);
  EXPECT_EQ(exS1.origin(), node->output(0));
  EXPECT_EQ(exS2.origin(), node->output(1));
  EXPECT_EQ(exS3.origin(), node->output(0));
  EXPECT_EQ(exS4.origin(), node->output(1));
  EXPECT_EQ(exS5.origin(), node->output(2));
}

TEST(StoreOperationTests, TestStoreAllocaReduction)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = TestType::createValueType();
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
  EXPECT_TRUE(success1 && success2);
  bool has_add_import = false;
  for (size_t n = 0; n < graph.GetRootRegion().nresults(); n++)
  {
    if (graph.GetRootRegion().result(n)->origin() == s)
      has_add_import = true;
  }
  EXPECT_TRUE(has_add_import);
}

TEST(StoreOperationTests, TestStoreStoreReduction)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = TestType::createValueType();
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
  EXPECT_TRUE(success);
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 1u);
  EXPECT_EQ(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin())->input(1)->origin(), v2);
}

TEST(StoreOperationTests, IOBarrierAllocaAddressNormalization)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();
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
  EXPECT_TRUE(successStoreNode1);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex1.origin())->input(0)->origin(),
      allocaResults[0]);

  // There is no IOBarrierOperation node as producer for the store address. We expect the
  // normalization not to trigger.
  EXPECT_FALSE(successStoreNode2);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex2.origin())->input(0)->origin(),
      addressImport);
}

TEST(StoreOperationTests, IOBarrierAllocaAddressNormalization_Gamma)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();
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
  EXPECT_TRUE(successStoreNode);
  // There should only be the store node left.
  // The IOBarrier node should have been pruned.
  EXPECT_EQ(gammaNode->subregion(0)->numNodes(), 1u);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*exitVar.branchResult[0]->origin())
          ->input(0)
          ->origin(),
      addressEntryVar.branchArgument[0]);
}

TEST(StoreOperationTests, storeAllocaSingleUser)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();
  const auto bit32Type = BitType::Create(32);

  Graph graph;
  const auto valueImport = &jlm::rvsdg::GraphImport::Create(graph, valueType, "value");
  const auto sizeImport = &jlm::rvsdg::GraphImport::Create(graph, bit32Type, "value");

  auto allocaResults = AllocaOperation::create(valueType, sizeImport, 4);

  auto & storeNode = StoreNonVolatileOperation::CreateNode(
      *allocaResults[0],
      *valueImport,
      { allocaResults[1] },
      4);

  auto & x1 = GraphExport::Create(*storeNode.output(0), "store");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::normalizeStoreAllocaSingleUser,
      storeNode);

  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_TRUE(success);
  EXPECT_EQ(x1.origin(), allocaResults[1]);
}

TEST(StoreOperationTests, storeAllocaSingleUser_MultipleUsers)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();
  const auto bit32Type = BitType::Create(32);

  Graph graph;
  const auto valueImport = &jlm::rvsdg::GraphImport::Create(graph, valueType, "value");
  const auto sizeImport = &jlm::rvsdg::GraphImport::Create(graph, bit32Type, "value");

  auto allocaResults = AllocaOperation::create(valueType, sizeImport, 4);

  auto & storeNode1 = StoreNonVolatileOperation::CreateNode(
      *allocaResults[0],
      *valueImport,
      { allocaResults[1] },
      4);

  auto & storeNode2 = StoreNonVolatileOperation::CreateNode(
      *allocaResults[0],
      *valueImport,
      { storeNode1.output(0) },
      4);

  auto & x1 = GraphExport::Create(*storeNode1.output(0), "store1");
  auto & x2 = GraphExport::Create(*storeNode2.output(0), "store2");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto successStoreNode1 = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::normalizeStoreAllocaSingleUser,
      storeNode1);
  const auto successStoreNode2 = jlm::rvsdg::ReduceNode<StoreNonVolatileOperation>(
      StoreNonVolatileOperation::normalizeStoreAllocaSingleUser,
      storeNode2);

  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_FALSE(successStoreNode1);
  EXPECT_FALSE(successStoreNode2);
  EXPECT_EQ(x1.origin(), storeNode1.output(0));
  EXPECT_EQ(x2.origin(), storeNode2.output(0));
}
