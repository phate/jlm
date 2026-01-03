/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(ControlFlowGraphPruneTests, test)
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

  EXPECT_EQ(cfg.nnodes(), 1);
}
