/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/frontend/ControlFlowRestructuring.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>

TEST(ControlFlowRestructuringTests, AcyclicStructured)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);
  auto bb3 = BasicBlock::create(cfg);
  auto bb4 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(bb2);
  bb1->add_outedge(bb3);
  bb2->add_outedge(bb4);
  bb3->add_outedge(bb4);
  bb4->add_outedge(cfg.exit());

  //	jlm::view_ascii(cfg, stdout);

  size_t nnodes = cfg.nnodes();
  RestructureBranches(cfg);

  //	jlm::view_ascii(cfg, stdout);

  EXPECT_EQ(nnodes, cfg.nnodes());
}

TEST(ControlFlowRestructuringTests, AcyclicUnstructured)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);
  auto bb3 = BasicBlock::create(cfg);
  auto bb4 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(bb2);
  bb1->add_outedge(bb3);
  bb2->add_outedge(bb3);
  bb2->add_outedge(bb4);
  bb3->add_outedge(bb4);
  bb4->add_outedge(cfg.exit());

  //	jlm::view_ascii(cfg, stdout);

  RestructureBranches(cfg);

  //	jlm::view_ascii(cfg, stdout);

  EXPECT_TRUE(is_proper_structured(cfg));
}

TEST(ControlFlowRestructuringTests, DoWhileLoop)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);
  auto bb3 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(bb2);
  bb2->add_outedge(bb2);
  bb2->add_outedge(bb3);
  bb3->add_outedge(bb1);
  bb3->add_outedge(cfg.exit());

  //	jlm::view_ascii(cfg, stdout);

  size_t nnodes = cfg.nnodes();
  RestructureControlFlow(cfg);

  //	jlm::view_ascii(cfg, stdout);

  EXPECT_EQ(nnodes, cfg.nnodes());
  EXPECT_EQ(bb2->OutEdge(0)->sink(), bb2);
  EXPECT_EQ(bb3->OutEdge(0)->sink(), bb1);
}

TEST(ControlFlowRestructuringTests, WhileLoop)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(cfg.exit());
  bb1->add_outedge(bb2);
  bb2->add_outedge(bb1);

  //	jlm::view_ascii(cfg, stdout);

  RestructureControlFlow(cfg);

  /* FIXME: Nodes are not printed in the right order */
  //	jlm::view_ascii(cfg, stdout);

  EXPECT_TRUE(is_proper_structured(cfg));
}

TEST(ControlFlowRestructuringTests, IrreducibleCfg)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);
  auto bb3 = BasicBlock::create(cfg);
  auto bb4 = BasicBlock::create(cfg);
  auto bb5 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(bb2);
  bb1->add_outedge(bb3);
  bb2->add_outedge(bb4);
  bb2->add_outedge(bb3);
  bb3->add_outedge(bb2);
  bb3->add_outedge(bb5);
  bb4->add_outedge(cfg.exit());
  bb5->add_outedge(cfg.exit());

  //	jlm::view_ascii(cfg, stdout);

  RestructureControlFlow(cfg);

  //	jlm::view_ascii(cfg, stdout);
  EXPECT_TRUE(is_proper_structured(cfg));
}

TEST(ControlFlowRestructuringTests, AcyclicUnstructuredInDoWhileLoop)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);
  auto bb3 = BasicBlock::create(cfg);
  auto bb4 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(bb3);
  bb1->add_outedge(bb2);
  bb2->add_outedge(bb3);
  bb2->add_outedge(bb4);
  bb3->add_outedge(bb4);
  bb4->add_outedge(bb1);
  bb4->add_outedge(cfg.exit());

  //	jlm::view_ascii(cfg, stdout);

  RestructureControlFlow(cfg);

  //	jlm::view_ascii(cfg, stdout);
  EXPECT_TRUE(is_proper_structured(cfg));
}

TEST(ControlFlowRestructuringTests, LorBeforeDoWhileLoop)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);
  auto bb3 = BasicBlock::create(cfg);
  auto bb4 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(bb2);
  bb1->add_outedge(bb3);
  bb2->add_outedge(bb4);
  bb2->add_outedge(bb3);
  bb3->add_outedge(bb4);
  bb4->add_outedge(cfg.exit());
  bb4->add_outedge(bb4);

  //	jlm::view_ascii(cfg, stdout);

  RestructureControlFlow(cfg);

  //	jlm::view_ascii(cfg, stdout);
  EXPECT_TRUE(is_proper_structured(cfg));
}

TEST(ControlFlowRestructuringTests, StaticEndlessLoop)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(im);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(bb2);
  bb1->add_outedge(bb1);
  bb1->add_outedge(cfg.exit());
  bb2->add_outedge(bb2);

  //	jlm::print_dot(cfg, stdout);

  RestructureControlFlow(cfg);

  //	jlm::print_dot(cfg, stdout);
  EXPECT_TRUE(is_proper_structured(cfg));
}
