/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <jlm/rvsdg/view.hpp>

static void
test_load_mux_reduction()
{
  using namespace jlm;

  jlm::valuetype vt;
  PointerType pt;
  MemoryStateType mt;

  jive::graph graph;
  auto nf = LoadOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_mux_reducible(false);

  auto a = graph.add_import({pt, "a"});
  auto s1 = graph.add_import({mt, "s1"});
  auto s2 = graph.add_import({mt, "s2"});
  auto s3 = graph.add_import({mt, "s3"});

  auto mux = MemStateMergeOperator::Create({s1, s2, s3});
  auto ld = LoadNode::Create(a, {mux}, vt, 4);

  auto ex1 = graph.add_export(ld[0], {ld[0]->type(), "v"});
  auto ex2 = graph.add_export(ld[1], {ld[1]->type(), "s"});

  // jive::view(graph.root(), stdout);

  nf->set_mutable(true);
  nf->set_load_mux_reducible(true);
  graph.normalize();
  graph.prune();

  // jive::view(graph.root(), stdout);

  auto load = jive::node_output::node(ex1->origin());
  assert(is<LoadOperation>(load));
  assert(load->ninputs() == 4);
  assert(load->input(1)->origin() == s1);
  assert(load->input(2)->origin() == s2);
  assert(load->input(3)->origin() == s3);

  auto merge = jive::node_output::node(ex2->origin());
  assert(is<jlm::MemStateMergeOperator>(merge));
  assert(merge->ninputs() == 3);
  for (size_t n = 0; n < merge->ninputs(); n++) {
    auto node = jive::node_output::node(merge->input(n)->origin());
    assert(node == load);
  }
}

static void
test_load_mux_reduction2()
{
  /*
   * Arrange
   */
  using namespace jlm;

  jlm::valuetype vt;
  PointerType pt;
  MemoryStateType mt;

  jive::graph graph;
  auto nf = LoadOperation::GetNormalForm(&graph);
  nf->set_mutable(false);
  nf->set_load_mux_reducible(false);

  auto a = graph.add_import({pt, "a"});
  auto s1 = graph.add_import({mt, "s1"});
  auto s2 = graph.add_import({mt, "s2"});

  auto merge = MemStateMergeOperator::Create(std::vector<jive::output*>{s1, s2});
  auto ld = LoadNode::Create(a, {merge, merge}, vt, 4);

  auto ex1 = graph.add_export(ld[0], {ld[0]->type(), "v"});
  auto ex2 = graph.add_export(ld[1], {ld[1]->type(), "s1"});
  auto ex3 = graph.add_export(ld[2], {ld[2]->type(), "s2"});

  jive::view(graph.root(), stdout);

  /*
   * Act
   */
  nf->set_mutable(true);
  nf->set_load_mux_reducible(true);
  graph.normalize();
  graph.prune();

  jive::view(graph.root(), stdout);

  /*
   * Assert
   *
   * The LoadMux reduction should not be performed, as the current implementation does not correctly take care of
   * the two identical load state operands originating from the merge node.
   */
  assert(ld.size() == 3);
  assert(ex1->origin() == ld[0]);
  assert(ex2->origin() == ld[1]);
  assert(ex3->origin() == ld[2]);
}

static int
test()
{
  test_load_mux_reduction();
  test_load_mux_reduction2();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/TestLoadMuxReduction", test)