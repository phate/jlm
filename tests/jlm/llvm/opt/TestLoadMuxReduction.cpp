/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

#include <jlm/rvsdg/view.hpp>

static void
TestSuccess()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_mux_reducible(false);

  auto a = &jlm::tests::GraphImport::Create(graph, pt, "a");
  auto s1 = &jlm::tests::GraphImport::Create(graph, mt, "s1");
  auto s2 = &jlm::tests::GraphImport::Create(graph, mt, "s2");
  auto s3 = &jlm::tests::GraphImport::Create(graph, mt, "s3");

  auto mux = MemoryStateMergeOperation::Create({ s1, s2, s3 });
  auto ld = LoadNonVolatileNode::Create(a, { mux }, vt, 4);

  auto & ex1 = GraphExport::Create(*ld[0], "v");
  auto & ex2 = GraphExport::Create(*ld[1], "s");

  // jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_load_mux_reducible(true);
  graph.normalize();
  graph.prune();

  // jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto load = jlm::rvsdg::output::GetNode(*ex1.origin());
  assert(is<LoadNonVolatileOperation>(load));
  assert(load->ninputs() == 4);
  assert(load->input(1)->origin() == s1);
  assert(load->input(2)->origin() == s2);
  assert(load->input(3)->origin() == s3);

  auto merge = jlm::rvsdg::output::GetNode(*ex2.origin());
  assert(is<MemoryStateMergeOperation>(merge));
  assert(merge->ninputs() == 3);
  for (size_t n = 0; n < merge->ninputs(); n++)
  {
    auto node = jlm::rvsdg::output::GetNode(*merge->input(n)->origin());
    assert(node == load);
  }
}

static void
TestWrongNumberOfOperands()
{
  // Arrange
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_mux_reducible(false);

  auto a = &jlm::tests::GraphImport::Create(graph, pt, "a");
  auto s1 = &jlm::tests::GraphImport::Create(graph, mt, "s1");
  auto s2 = &jlm::tests::GraphImport::Create(graph, mt, "s2");

  auto merge = MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>{ s1, s2 });
  auto ld = LoadNonVolatileNode::Create(a, { merge, merge }, vt, 4);

  auto & ex1 = GraphExport::Create(*ld[0], "v");
  auto & ex2 = GraphExport::Create(*ld[1], "s1");
  auto & ex3 = GraphExport::Create(*ld[2], "s2");

  jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_load_mux_reducible(true);
  graph.normalize();
  graph.prune();

  jlm::rvsdg::view(graph.root(), stdout);

  // Assert

  // The LoadMux reduction should not be performed, as the current implementation does not correctly
  // take care of the two identical load state operands originating from the merge node.
  assert(ld.size() == 3);
  assert(ex1.origin() == ld[0]);
  assert(ex2.origin() == ld[1]);
  assert(ex3.origin() == ld[2]);
}

static void
TestLoadWithoutStates()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();

  jlm::rvsdg::graph graph;
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_mux_reducible(false);

  auto address = &jlm::tests::GraphImport::Create(graph, pointerType, "address");

  auto loadResults = LoadNonVolatileNode::Create(address, {}, valueType, 4);

  auto & ex = GraphExport::Create(*loadResults[0], "v");

  jlm::rvsdg::view(graph.root(), stdout);

  // Act
  nf->set_mutable(true);
  nf->set_load_mux_reducible(true);
  graph.normalize();
  graph.prune();

  jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  auto load = jlm::rvsdg::output::GetNode(*ex.origin());
  assert(is<LoadNonVolatileOperation>(load));
  assert(load->ninputs() == 1);
}

static int
TestLoadMuxReduction()
{
  TestSuccess();
  TestWrongNumberOfOperands();
  TestLoadWithoutStates();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/TestLoadMuxReduction", TestLoadMuxReduction)
