/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(ControlFlowGraphValidityTests, test_single_operand_phi)
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
