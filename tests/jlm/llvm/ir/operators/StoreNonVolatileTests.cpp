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
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StoreNonVolatile.hpp>

static int
OperationEquality()
{
  using namespace jlm::llvm;

  // Arrange
  MemoryStateType memoryType;
  jlm::tests::valuetype valueType;
  PointerType pointerType;

  StoreNonVolatileOperation operation1(valueType, 2, 4);
  StoreNonVolatileOperation operation2(pointerType, 2, 4);
  StoreNonVolatileOperation operation3(valueType, 4, 4);
  StoreNonVolatileOperation operation4(valueType, 2, 8);
  jlm::tests::test_op operation5({ &pointerType }, { &pointerType });

  // Act & Assert
  assert(operation1 == operation1);
  assert(operation1 != operation2); // stored type differs
  assert(operation1 != operation3); // number of memory states differs
  assert(operation1 != operation4); // alignment differs
  assert(operation1 != operation5); // operation differs

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/operators/StoreNonVolatileTests-OperationEquality",
    OperationEquality)

static void
TestCopy()
{
  using namespace jlm::llvm;

  jlm::tests::valuetype valueType;
  PointerType pointerType;
  MemoryStateType memoryStateType;

  jlm::rvsdg::graph graph;
  auto address1 = graph.add_import({ pointerType, "address1" });
  auto value1 = graph.add_import({ valueType, "value1" });
  auto memoryState1 = graph.add_import({ memoryStateType, "state1" });

  auto address2 = graph.add_import({ pointerType, "address2" });
  auto value2 = graph.add_import({ valueType, "value2" });
  auto memoryState2 = graph.add_import({ memoryStateType, "state2" });

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
  jlm::tests::valuetype vt;
  PointerType pt;
  MemoryStateType mt;

  jlm::rvsdg::graph graph;
  auto nf = graph.node_normal_form(typeid(StoreNonVolatileOperation));
  auto snf = static_cast<jlm::llvm::store_normal_form *>(nf);
  snf->set_mutable(false);
  snf->set_store_mux_reducible(false);

  auto a = graph.add_import({ pt, "a" });
  auto v = graph.add_import({ vt, "v" });
  auto s1 = graph.add_import({ mt, "s1" });
  auto s2 = graph.add_import({ mt, "s2" });
  auto s3 = graph.add_import({ mt, "s3" });

  auto mux = MemStateMergeOperator::Create({ s1, s2, s3 });
  auto state = StoreNonVolatileNode::Create(a, v, { mux }, 4);

  auto ex = graph.add_export(state[0], { state[0]->type(), "s" });

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  snf->set_mutable(true);
  snf->set_store_mux_reducible(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto muxnode = jlm::rvsdg::node_output::node(ex->origin());
  assert(is<MemStateMergeOperator>(muxnode));
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
  jlm::tests::valuetype vt;
  PointerType pt;
  MemoryStateType mt;

  jlm::rvsdg::graph graph;
  auto nf = graph.node_normal_form(typeid(StoreNonVolatileOperation));
  auto snf = static_cast<jlm::llvm::store_normal_form *>(nf);
  snf->set_mutable(false);
  snf->set_multiple_origin_reducible(false);

  auto a = graph.add_import({ pt, "a" });
  auto v = graph.add_import({ vt, "v" });
  auto s = graph.add_import({ mt, "s" });

  auto states = StoreNonVolatileNode::Create(a, v, { s, s, s, s }, 4);

  auto ex = graph.add_export(states[0], { states[0]->type(), "s" });

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  snf->set_mutable(true);
  snf->set_multiple_origin_reducible(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto node = jlm::rvsdg::node_output::node(ex->origin());
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(node->operation()) && node->ninputs() == 3);
}

static void
TestStoreAllocaReduction()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::tests::valuetype vt;
  MemoryStateType mt;
  jlm::rvsdg::bittype bt(32);

  jlm::rvsdg::graph graph;
  auto nf = graph.node_normal_form(typeid(StoreNonVolatileOperation));
  auto snf = static_cast<jlm::llvm::store_normal_form *>(nf);
  snf->set_mutable(false);
  snf->set_store_alloca_reducible(false);

  auto size = graph.add_import({ bt, "size" });
  auto value = graph.add_import({ vt, "value" });
  auto s = graph.add_import({ mt, "s" });

  auto alloca1 = alloca_op::create(vt, size, 4);
  auto alloca2 = alloca_op::create(vt, size, 4);
  auto states1 = StoreNonVolatileNode::Create(alloca1[0], value, { alloca1[1], alloca2[1], s }, 4);
  auto states2 = StoreNonVolatileNode::Create(alloca2[0], value, states1, 4);

  graph.add_export(states2[0], { states2[0]->type(), "s1" });
  graph.add_export(states2[1], { states2[1]->type(), "s2" });
  graph.add_export(states2[2], { states2[2]->type(), "s3" });

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
  jlm::tests::valuetype vt;
  PointerType pt;
  MemoryStateType mt;

  jlm::rvsdg::graph graph;
  auto a = graph.add_import({ pt, "address" });
  auto v1 = graph.add_import({ vt, "value" });
  auto v2 = graph.add_import({ vt, "value" });
  auto s = graph.add_import({ mt, "state" });

  auto s1 = StoreNonVolatileNode::Create(a, v1, { s }, 4)[0];
  auto s2 = StoreNonVolatileNode::Create(a, v2, { s1 }, 4)[0];

  auto ex = graph.add_export(s2, { s2->type(), "state" });

  jlm::rvsdg::view(graph.root(), stdout);

  // Act
  auto nf = StoreNonVolatileOperation::GetNormalForm(&graph);
  nf->set_store_store_reducible(true);
  graph.normalize();
  graph.prune();

  jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  assert(graph.root()->nnodes() == 1);
  assert(jlm::rvsdg::node_output::node(ex->origin())->input(1)->origin() == v2);
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

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/StoreNonVolatileTests", TestStore)
