/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/Gamma.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::llvm
{

TEST(GammaTests, test_predicate_reduction)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  BitType bits2(2);

  auto v0 = &GraphImport::Create(graph, BitType::Create(32), "");
  auto v1 = &GraphImport::Create(graph, BitType::Create(32), "");
  auto v2 = &GraphImport::Create(graph, BitType::Create(32), "");

  auto pred = &ControlConstantOperation::create(graph.GetRootRegion(), 3, 1);

  auto gammaNode = GammaNode::create(pred, 3);
  auto ev0 = gammaNode->AddEntryVar(v0);
  auto ev1 = gammaNode->AddEntryVar(v1);
  auto ev2 = gammaNode->AddEntryVar(v2);
  gammaNode->AddExitVar({ ev0.branchArgument[0], ev1.branchArgument[1], ev2.branchArgument[2] });

  auto & r = GraphExport::Create(*gammaNode->output(0), "");

  view(&graph.GetRootRegion(), stdout);

  // Act
  reduceStaticallyKnownPredicate(*gammaNode);
  view(&graph.GetRootRegion(), stdout);

  // Assert
  EXPECT_EQ(r.origin(), v1);

  graph.PruneNodes();
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 0u);
}

}
