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
#include <jlm/rvsdg/NodeNormalization.hpp>

static void
test_bitunary_reduction()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bt32 = bittype::Create(32);

  auto nf = sext_op::normal_form(&graph);
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, bt32, "x");

  auto y = bitnot_op::create(32, x);
  auto sextNode = jlm::rvsdg::output::GetNode(*sext_op::create(64, y));

  auto & ex = jlm::llvm::GraphExport::Create(*sextNode->output(0), "x");

  view(graph, stdout);

  // Act
  ReduceNode<sext_op>(NormalizeUnaryOperation, *sextNode);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(is<bitnot_op>(jlm::rvsdg::output::GetNode(*ex.origin())));
}

static inline void
test_bitbinary_reduction()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bt32 = bittype::Create(32);

  auto nf = sext_op::normal_form(&graph);
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, bt32, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bt32, "y");

  auto z = bitadd_op::create(32, x, y);
  auto sextNode = jlm::rvsdg::output::GetNode(*sext_op::create(64, z));

  auto & ex = jlm::llvm::GraphExport::Create(*sextNode->output(0), "x");

  view(graph, stdout);

  // Act
  ReduceNode<sext_op>(NormalizeUnaryOperation, *sextNode);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(jlm::rvsdg::is<jlm::rvsdg::bitadd_op>(jlm::rvsdg::output::GetNode(*ex.origin())));
}

static void
test_inverse_reduction()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  Graph graph;
  auto bt64 = bittype::Create(64);

  auto nf = sext_op::normal_form(&graph);
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, bt64, "x");

  auto y = trunc_op::create(32, x);
  auto sextNode = jlm::rvsdg::output::GetNode(*sext_op::create(64, y));

  auto & ex = jlm::llvm::GraphExport::Create(*sextNode->output(0), "x");

  view(graph, stdout);

  // Act
  ReduceNode<sext_op>(NormalizeUnaryOperation, *sextNode);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
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
