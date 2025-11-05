/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

static void
test()
{
  using namespace jlm::llvm;

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");

  ControlFlowGraph cfg(module);
  auto bb0 = BasicBlock::create(cfg);
  auto bb1 = BasicBlock::create(cfg);

  jlm::rvsdg::ControlConstantOperation op(jlm::rvsdg::ControlValueRepresentation(1, 2));
  bb0->append_last(ThreeAddressCode::create(op, {}));
  bb0->append_last(BranchOperation::create(2, bb0->last()->result(0)));

  cfg.exit()->divert_inedges(bb0);
  bb0->add_outedge(bb1);
  bb0->add_outedge(cfg.exit());
  bb1->add_outedge(bb1);

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;

  purge(cfg);

  assert(cfg.nnodes() == 2);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-cfg-purge", test)
