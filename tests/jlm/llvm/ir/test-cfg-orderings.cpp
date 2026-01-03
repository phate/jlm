/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>

TEST(ControlFlowGraphOrderTests, test)
{
  using namespace jlm::llvm;

  // Arrange
  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(im);
  auto bb0 = BasicBlock::create(cfg);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);
  auto bb3 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb0);
  bb0->add_outedge(bb1);
  bb0->add_outedge(bb2);
  bb1->add_outedge(bb3);
  bb2->add_outedge(bb3);
  bb3->add_outedge(cfg.exit());

  // check orderings
  std::vector<ControlFlowGraphNode *> po1({ cfg.exit(), bb3, bb2, bb1, bb0, cfg.entry() });
  std::vector<ControlFlowGraphNode *> po2({ cfg.exit(), bb3, bb1, bb2, bb0, cfg.entry() });
  EXPECT_TRUE(postorder(cfg) == po1 || postorder(cfg) == po2);

  std::vector<ControlFlowGraphNode *> rpo1({ cfg.entry(), bb0, bb1, bb2, bb3, cfg.exit() });
  std::vector<ControlFlowGraphNode *> rpo2({ cfg.entry(), bb0, bb2, bb1, bb3, cfg.exit() });
  EXPECT_TRUE(reverse_postorder(cfg) == rpo1 || reverse_postorder(cfg) == rpo2);
}
