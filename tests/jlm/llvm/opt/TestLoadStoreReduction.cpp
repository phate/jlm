/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
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
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_store_reducible(false);

  auto address = graph.add_import({ pointerType, "address" });
  auto value = graph.add_import({ jlm::rvsdg::bittype::Create(32), "value" });
  auto memoryState = graph.add_import({ memoryStateType, "memoryState" });

  auto storeResults = StoreNonVolatileNode::Create(address, value, { memoryState }, 4);
  auto loadResults =
      LoadNonVolatileNode::Create(address, storeResults, *jlm::rvsdg::bittype::Create(8), 4);

  auto exportedValue = graph.add_export(loadResults[0], { jlm::rvsdg::bittype::Create(8), "v" });
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
  assert(is<LoadNonVolatileOperation>(load));
  assert(load->ninputs() == 2);
  auto store = jlm::rvsdg::node_output::node(load->input(1)->origin());
  assert(is<StoreNonVolatileOperation>(store));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/TestLoadStoreReductionWithDifferentValueOperandType",
    TestLoadStoreReductionWithDifferentValueOperandType)
