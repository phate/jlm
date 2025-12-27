/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/ssa.hpp>
#include <jlm/rvsdg/TestType.hpp>

static inline void
test_two_phis()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;

  auto vt = jlm::rvsdg::TestType::createValueType();
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

  bb2->append_last(ThreeAddressCode::create(TestOperation::create({}, { vt }), {}));
  auto v1 = bb2->last()->result(0);

  bb2->append_last(ThreeAddressCode::create(TestOperation::create({}, { vt }), {}));
  auto v3 = bb2->last()->result(0);

  bb3->append_last(ThreeAddressCode::create(TestOperation::create({}, { vt }), {}));
  auto v2 = bb3->last()->result(0);

  bb3->append_last(ThreeAddressCode::create(TestOperation::create({}, { vt }), {}));
  auto v4 = bb3->last()->result(0);

  bb4->append_last(SsaPhiOperation::create({ { v1, bb2 }, { v2, bb3 } }, vt));
  bb4->append_last(SsaPhiOperation::create({ { v3, bb2 }, { v4, bb3 } }, vt));

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;

  destruct_ssa(cfg);

  std::cout << ControlFlowGraph::ToAscii(cfg) << std::flush;
}

static void
verify()
{
  test_two_phis();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-ssa-destruction", verify)
