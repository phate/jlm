/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/ControlFlowGraph.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

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
  EXPECT_EQ(cfg.nnodes(), 0u);
}

TEST(ControlFlowGraphTests, test_straightening)
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

TEST(ControLFlowGraphTests, testPurge)
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb0 = BasicBlock::create(cfg);
  auto bb1 = BasicBlock::create(cfg);

  auto op = std::make_unique<jlm::rvsdg::ControlConstantOperation>(
      jlm::rvsdg::ControlValueRepresentation(1, 2));
  bb0->append_last(ThreeAddressCode::create(std::move(op), {}));
  bb0->append_last(BranchOperation::create(2, bb0->last()->result(0)));

  cfg.exit()->divert_inedges(bb0);
  bb0->add_outedge(bb1);
  bb0->add_outedge(cfg.exit());
  bb1->add_outedge(bb1);

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;

  purge(cfg);

  EXPECT_EQ(cfg.nnodes(), 2u);
}

TEST(ControlFlowGraphTests, test_is_structured)
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

TEST(ControlFlowGraphTests, testPrune)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();

  // Arrange
  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(im);
  auto arg = cfg.entry()->append_argument(Argument::create("arg", vt));
  auto bb0 = BasicBlock::create(cfg);
  auto bb1 = BasicBlock::create(cfg);

  bb0->append_last(ThreeAddressCode::create(TestOperation::create({}, { vt }), {}));
  bb1->append_last(
      SsaPhiOperation::create({ { bb0->last()->result(0), bb0 }, { arg, cfg.entry() } }, vt));

  cfg.exit()->divert_inedges(bb1);
  bb0->add_outedge(bb1);
  bb1->add_outedge(cfg.exit());
  cfg.exit()->append_result(bb1->last()->result(0));

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;

  /* verify pruning */

  prune(cfg);
  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;

  EXPECT_EQ(cfg.nnodes(), 1u);
}

TEST(ControlFlowGraphTests, testOrdering)
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

TEST(ControlFlowGraphTests, test_divert_inedges)
{
  using namespace jlm::llvm;

  // Arrange
  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(im);

  auto bb0 = BasicBlock::create(cfg);

  cfg.exit()->divert_inedges(bb0);
  bb0->add_outedge(bb0);
  bb0->add_outedge(cfg.exit());

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;

  /* verify inedge diversion */

  bb0->divert_inedges(bb0);
}

TEST(ControlFlowGraphTests, testIsValid_SingleOperandPhi)
{
  using namespace jlm::llvm;

  auto vt = jlm::rvsdg::TestType::createValueType();

  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(im);
  auto arg = cfg.entry()->append_argument(Argument::create("arg", vt));

  auto bb0 = BasicBlock::create(cfg);
  bb0->append_first(SsaPhiOperation::create({ { arg, cfg.entry() } }, vt));

  cfg.exit()->divert_inedges(bb0);
  bb0->add_outedge(cfg.exit());
  cfg.exit()->append_result(bb0->last()->result(0));

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;

  EXPECT_TRUE(is_valid(cfg));
}
