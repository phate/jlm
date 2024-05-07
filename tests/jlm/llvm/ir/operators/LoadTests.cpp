/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/statemux.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>

static int
OperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;
  PointerType pointerType;

  LoadNonVolatileOperation operation1(valueType, 2, 4);
  LoadNonVolatileOperation operation2(pointerType, 2, 4);
  LoadNonVolatileOperation operation3(valueType, 4, 4);
  LoadNonVolatileOperation operation4(valueType, 2, 8);
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
    "jlm/llvm/ir/operators/LoadNonVolatileTests-OperationEquality",
    OperationEquality)

static void
TestCopy()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;
  PointerType pointerType;

  jlm::rvsdg::graph graph;
  auto address1 = graph.add_import({ pointerType, "address1" });
  auto memoryState1 = graph.add_import({ memoryType, "memoryState1" });

  auto address2 = graph.add_import({ pointerType, "address2" });
  auto memoryState2 = graph.add_import({ memoryType, "memoryState2" });

  auto loadResults = LoadNonVolatileNode::Create(address1, { memoryState1 }, valueType, 4);

  // Act
  auto node = jlm::rvsdg::node_output::node(loadResults[0]);
  auto loadNode = jlm::util::AssertedCast<const LoadNonVolatileNode>(node);
  auto copiedNode = loadNode->copy(graph.root(), { address2, memoryState2 });

  // Assert
  auto copiedLoadNode = dynamic_cast<const LoadNonVolatileNode *>(copiedNode);
  assert(copiedLoadNode != nullptr);
  assert(loadNode->GetOperation() == copiedLoadNode->GetOperation());
}

static void
TestLoadAllocaReduction()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType mt;
  jlm::rvsdg::bittype bt(32);

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_alloca_reducible(false);

  auto size = graph.add_import({ bt, "v" });

  auto alloca1 = alloca_op::create(bt, size, 4);
  auto alloca2 = alloca_op::create(bt, size, 4);
  auto mux = jlm::rvsdg::create_state_mux(mt, { alloca1[1] }, 1);
  auto value =
      LoadNonVolatileNode::Create(alloca1[0], { alloca1[1], alloca2[1], mux[0] }, bt, 4)[0];

  auto ex = graph.add_export(value, { value->type(), "l" });

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_load_alloca_reducible(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto node = jlm::rvsdg::node_output::node(ex->origin());
  assert(is<LoadNonVolatileOperation>(node));
  assert(node->ninputs() == 3);
  assert(node->input(1)->origin() == alloca1[1]);
  assert(node->input(2)->origin() == mux[0]);
}

static void
TestMultipleOriginReduction()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType mt;
  jlm::tests::valuetype vt;
  PointerType pt;

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_multiple_origin_reducible(false);

  auto a = graph.add_import({ pt, "a" });
  auto s = graph.add_import({ mt, "s" });

  auto load = LoadNonVolatileNode::Create(a, { s, s, s, s }, vt, 4)[0];

  auto ex = graph.add_export(load, { load->type(), "l" });

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_multiple_origin_reducible(true);
  graph.normalize();

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto node = jlm::rvsdg::node_output::node(ex->origin());
  assert(is<LoadNonVolatileOperation>(node));
  assert(node->ninputs() == 2);
}

static void
TestLoadStoreStateReduction()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::rvsdg::bittype bt(32);

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_store_state_reducible(false);

  auto size = graph.add_import({ bt, "v" });

  auto alloca1 = alloca_op::create(bt, size, 4);
  auto alloca2 = alloca_op::create(bt, size, 4);
  auto store1 = StoreNonVolatileNode::Create(alloca1[0], size, { alloca1[1] }, 4);
  auto store2 = StoreNonVolatileNode::Create(alloca2[0], size, { alloca2[1] }, 4);

  auto value1 = LoadNonVolatileNode::Create(alloca1[0], { store1[0], store2[0] }, bt, 4)[0];
  auto value2 = LoadNonVolatileNode::Create(alloca1[0], { store1[0] }, bt, 8)[0];

  auto ex1 = graph.add_export(value1, { value1->type(), "l1" });
  auto ex2 = graph.add_export(value2, { value2->type(), "l2" });

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_load_store_state_reducible(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto node = jlm::rvsdg::node_output::node(ex1->origin());
  assert(is<LoadNonVolatileOperation>(node));
  assert(node->ninputs() == 2);

  node = jlm::rvsdg::node_output::node(ex2->origin());
  assert(is<LoadNonVolatileOperation>(node));
  assert(node->ninputs() == 2);
}

static void
TestLoadStoreReduction()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::tests::valuetype vt;
  PointerType pt;
  MemoryStateType mt;

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_store_reducible(false);

  auto a = graph.add_import({ pt, "address" });
  auto v = graph.add_import({ vt, "value" });
  auto s = graph.add_import({ mt, "state" });

  auto s1 = StoreNonVolatileNode::Create(a, v, { s }, 4)[0];
  auto load = LoadNonVolatileNode::Create(a, { s1 }, vt, 4);

  auto x1 = graph.add_export(load[0], { load[0]->type(), "value" });
  auto x2 = graph.add_export(load[1], { load[1]->type(), "state" });

  // jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_load_store_reducible(true);
  graph.normalize();

  // jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  assert(graph.root()->nnodes() == 1);
  assert(x1->origin() == v);
  assert(x2->origin() == s1);
}

static void
TestLoadLoadReduction()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::tests::valuetype vt;
  PointerType pt;
  MemoryStateType mt;

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);

  auto a1 = graph.add_import({ pt, "a1" });
  auto a2 = graph.add_import({ pt, "a2" });
  auto a3 = graph.add_import({ pt, "a3" });
  auto a4 = graph.add_import({ pt, "a4" });
  auto v1 = graph.add_import({ vt, "v1" });
  auto s1 = graph.add_import({ mt, "s1" });
  auto s2 = graph.add_import({ mt, "s2" });

  auto st1 = StoreNonVolatileNode::Create(a1, v1, { s1 }, 4);
  auto ld1 = LoadNonVolatileNode::Create(a2, { s1 }, vt, 4);
  auto ld2 = LoadNonVolatileNode::Create(a3, { s2 }, vt, 4);

  auto ld3 = LoadNonVolatileNode::Create(a4, { st1[0], ld1[1], ld2[1] }, vt, 4);

  auto x1 = graph.add_export(ld3[1], { mt, "s" });
  auto x2 = graph.add_export(ld3[2], { mt, "s" });
  auto x3 = graph.add_export(ld3[3], { mt, "s" });

  jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_load_load_state_reducible(true);
  graph.normalize();
  graph.prune();

  jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  assert(graph.root()->nnodes() == 6);

  auto ld = jlm::rvsdg::node_output::node(x1->origin());
  assert(is<LoadNonVolatileOperation>(ld));

  auto mx1 = jlm::rvsdg::node_output::node(x2->origin());
  assert(is<MemStateMergeOperator>(mx1) && mx1->ninputs() == 2);
  assert(mx1->input(0)->origin() == ld1[1] || mx1->input(0)->origin() == ld->output(2));
  assert(mx1->input(1)->origin() == ld1[1] || mx1->input(1)->origin() == ld->output(2));

  auto mx2 = jlm::rvsdg::node_output::node(x3->origin());
  assert(is<MemStateMergeOperator>(mx2) && mx2->ninputs() == 2);
  assert(mx2->input(0)->origin() == ld2[1] || mx2->input(0)->origin() == ld->output(3));
  assert(mx2->input(1)->origin() == ld2[1] || mx2->input(1)->origin() == ld->output(3));
}

static int
TestLoad()
{
  TestCopy();

  TestLoadAllocaReduction();
  TestMultipleOriginReduction();
  TestLoadStoreStateReduction();
  TestLoadStoreReduction();
  TestLoadLoadReduction();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/LoadNonVolatileTests", TestLoad)

static int
LoadVolatileOperationEquality()
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
    LoadVolatileOperationEquality)

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
