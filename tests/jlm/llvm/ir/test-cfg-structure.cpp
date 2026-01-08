/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(ControlFlowGraphStructureTests, test_straightening)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb1 = BasicBlock::create(cfg);
  auto bb2 = BasicBlock::create(cfg);
  auto bb3 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb1);
  bb1->add_outedge(bb2);
  bb2->add_outedge(bb3);
  bb3->add_outedge(cfg.exit());

  auto arg = cfg.entry()->append_argument(Argument::create("arg", vt));
  bb1->append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { arg }));
  bb2->append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { arg }));
  bb3->append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { arg }));

  auto bb3_last = static_cast<const BasicBlock *>(bb3)->tacs().last();
  straighten(cfg);

  EXPECT_EQ(cfg.nnodes(), 1u);
  auto node = cfg.entry()->OutEdge(0)->sink();

  EXPECT_TRUE(is<BasicBlock>(node));
  auto & tacs = static_cast<const BasicBlock *>(node)->tacs();
  EXPECT_EQ(tacs.ntacs(), 3u);
  EXPECT_EQ(tacs.last(), bb3_last);
}

TEST(ControlFlowGraphStructureTests, test_is_structured)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto split = BasicBlock::create(cfg);
  auto bb = BasicBlock::create(cfg);
  auto join = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(split);
  split->add_outedge(join);
  split->add_outedge(bb);
  bb->add_outedge(join);
  join->add_outedge(cfg.exit());

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;
  EXPECT_TRUE(is_structured(cfg));
}
