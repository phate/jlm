/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

static int
test()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  jlm::tests::test_op op({}, { vt });

  /* setup cfg */

  ipgraph_module im(jlm::util::FilePath(""), "", "");

  jlm::llvm::cfg cfg(im);
  auto arg = cfg.entry()->append_argument(argument::create("arg", vt));
  auto bb0 = BasicBlock::create(cfg);
  auto bb1 = BasicBlock::create(cfg);

  bb0->append_last(tac::create(op, {}));
  bb1->append_last(
      SsaPhiOperation::create({ { bb0->last()->result(0), bb0 }, { arg, cfg.entry() } }, vt));

  cfg.exit()->divert_inedges(bb1);
  bb0->add_outedge(bb1);
  bb1->add_outedge(cfg.exit());
  cfg.exit()->append_result(bb1->last()->result(0));

  std::cout << cfg::ToAscii(cfg) << std::flush;

  /* verify pruning */

  prune(cfg);
  std::cout << cfg::ToAscii(cfg) << std::flush;

  assert(cfg.nnodes() == 1);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-cfg-prune", test)
