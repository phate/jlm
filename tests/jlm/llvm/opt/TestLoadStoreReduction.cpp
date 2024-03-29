/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/load.hpp>
#include <jlm/llvm/ir/operators/store.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/view.hpp>

/**
 * Tests the load-store reduction with the value type of the store being different than the
 * value type of the load.
 */
static int
TestLoadStoreReductionWithDifferentValueOperandType()
{
  using namespace jlm::llvm;

  // Arrange
  PointerType pointerType;
  MemoryStateType memoryStateType;

  jlm::rvsdg::graph graph;
  auto nf = LoadOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_store_reducible(false);

  auto address = graph.add_import({ pointerType, "address" });
  auto value = graph.add_import({ jlm::rvsdg::bit32, "value" });
  auto memoryState = graph.add_import({ memoryStateType, "memoryState" });

  auto storeResults = StoreNode::Create(address, value, { memoryState }, 4);
  auto loadResults = LoadNode::Create(address, storeResults, jlm::rvsdg::bit8, 4);

  auto exportedValue = graph.add_export(loadResults[0], { jlm::rvsdg::bit8, "v" });
  graph.add_export(loadResults[1], { memoryStateType, "s" });

  jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_load_store_reducible(true);
  graph.normalize();
  graph.prune();

  jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto load = jlm::rvsdg::node_output::node(exportedValue->origin());
  assert(is<LoadOperation>(load));
  assert(load->ninputs() == 2);
  auto store = jlm::rvsdg::node_output::node(load->input(1)->origin());
  assert(is<StoreOperation>(store));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/TestLoadStoreReductionWithDifferentValueOperandType",
    TestLoadStoreReductionWithDifferentValueOperandType)
