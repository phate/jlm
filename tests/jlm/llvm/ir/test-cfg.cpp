/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/print.hpp>

TEST(ControLFlowGraphTests, test_remove_node)
{
  using namespace jlm::llvm;

  // Arrange
  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(im);

  auto bb0 = BasicBlock::create(cfg);
  bb0->add_outedge(bb0);
  bb0->add_outedge(cfg.exit());

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;

  // Act
  cfg.remove_node(bb0);

  // Assert
  EXPECT_EQ(cfg.nnodes(), 0);
}
