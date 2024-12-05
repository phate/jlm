/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

static inline void
test_bitunary_reduction()
{
  auto bt32 = jlm::rvsdg::bittype::Create(32);

  jlm::rvsdg::Graph graph;
  auto nf = jlm::llvm::sext_op::normal_form(&graph);
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, bt32, "x");

  auto y = jlm::rvsdg::bitnot_op::create(32, x);
  auto z = jlm::llvm::sext_op::create(64, y);

  auto & ex = jlm::llvm::GraphExport::Create(*z, "x");

  // jlm::rvsdg::view(graph, stdout);

  nf->set_mutable(true);
  graph.normalize();
  graph.prune();

  // jlm::rvsdg::view(graph, stdout);

  assert(jlm::rvsdg::is<jlm::rvsdg::bitnot_op>(jlm::rvsdg::output::GetNode(*ex.origin())));
}

static inline void
test_bitbinary_reduction()
{
  auto bt32 = jlm::rvsdg::bittype::Create(32);

  jlm::rvsdg::Graph graph;
  auto nf = jlm::llvm::sext_op::normal_form(&graph);
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, bt32, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bt32, "y");

  auto z = jlm::rvsdg::bitadd_op::create(32, x, y);
  auto w = jlm::llvm::sext_op::create(64, z);

  auto & ex = jlm::llvm::GraphExport::Create(*w, "x");

  //	jlm::rvsdg::view(graph, stdout);

  nf->set_mutable(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph, stdout);

  assert(jlm::rvsdg::is<jlm::rvsdg::bitadd_op>(jlm::rvsdg::output::GetNode(*ex.origin())));
}

static inline void
test_inverse_reduction()
{
  using namespace jlm;

  auto bt64 = jlm::rvsdg::bittype::Create(64);

  rvsdg::Graph graph;
  auto nf = jlm::llvm::sext_op::normal_form(&graph);
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, bt64, "x");

  auto y = jlm::llvm::trunc_op::create(32, x);
  auto z = jlm::llvm::sext_op::create(64, y);

  auto & ex = jlm::llvm::GraphExport::Create(*z, "x");

  jlm::rvsdg::view(graph, stdout);

  nf->set_mutable(true);
  graph.normalize();
  graph.prune();

  jlm::rvsdg::view(graph, stdout);

  assert(ex.origin() == x);
}

static int
test()
{
  test_bitunary_reduction();
  test_bitbinary_reduction();
  test_inverse_reduction();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/test-sext", test)
