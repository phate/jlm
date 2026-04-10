/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>

TEST(SExtOperationTests, test_bitunary_reduction)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bitType32 = BitType::Create(32);

  auto x = &jlm::rvsdg::GraphImport::Create(graph, bitType32, "x");

  auto y = bitnot_op::create(32, x);
  auto z = jlm::llvm::SExtOperation::create(64, y);

  auto & ex = GraphExport::Create(*z, "x");

  view(graph, stdout);

  // Act
  ReduceNode<SExtOperation>(
      NormalizeUnaryOperation,
      *jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*ex.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  EXPECT_TRUE(is<bitnot_op>(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin())));
}

TEST(SExtOperationTests, test_bitbinary_reduction)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bt32 = BitType::Create(32);

  auto x = &jlm::rvsdg::GraphImport::Create(graph, bt32, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, bt32, "y");

  auto z = bitadd_op::create(32, x, y);
  auto w = SExtOperation::create(64, z);

  auto & ex = GraphExport::Create(*w, "x");

  view(graph, stdout);

  // Act
  ReduceNode<SExtOperation>(
      NormalizeUnaryOperation,
      *jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*ex.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  EXPECT_TRUE(is<bitadd_op>(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin())));
}

TEST(SExtOperationTests, test_inverse_reduction)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bt64 = BitType::Create(64);

  auto x = &jlm::rvsdg::GraphImport::Create(graph, bt64, "x");

  auto y = TruncOperation::create(32, x);
  auto z = SExtOperation::create(64, y);

  auto & ex = GraphExport::Create(*z, "x");

  view(graph, stdout);

  // Act
  ReduceNode<SExtOperation>(
      NormalizeUnaryOperation,
      *jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*ex.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  EXPECT_EQ(ex.origin(), x);
}
