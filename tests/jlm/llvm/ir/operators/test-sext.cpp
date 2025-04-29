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
  auto bitType32 = bittype::Create(32);

  auto x = &jlm::tests::GraphImport::Create(graph, bitType32, "x");

  auto y = bitnot_op::create(32, x);
  auto z = jlm::llvm::sext_op::create(64, y);

  auto & ex = jlm::llvm::GraphExport::Create(*z, "x");

  view(graph, stdout);

  // Act
  ReduceNode<sext_op>(NormalizeUnaryOperation, *jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(is<bitnot_op>(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin())));
}

static void
test_bitbinary_reduction()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bt32 = bittype::Create(32);

  auto x = &jlm::tests::GraphImport::Create(graph, bt32, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bt32, "y");

  auto z = bitadd_op::create(32, x, y);
  auto w = sext_op::create(64, z);

  auto & ex = jlm::llvm::GraphExport::Create(*w, "x");

  view(graph, stdout);

  // Act
  ReduceNode<sext_op>(NormalizeUnaryOperation, *jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(is<bitadd_op>(jlm::rvsdg::output::GetNode(*ex.origin())));
}

static void
test_inverse_reduction()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bt64 = bittype::Create(64);

  auto x = &jlm::tests::GraphImport::Create(graph, bt64, "x");

  auto y = TruncOperation::create(32, x);
  auto z = sext_op::create(64, y);

  auto & ex = jlm::llvm::GraphExport::Create(*z, "x");

  view(graph, stdout);

  // Act
  ReduceNode<sext_op>(NormalizeUnaryOperation, *jlm::rvsdg::TryGetOwnerNode<Node>(*ex.origin()));
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
