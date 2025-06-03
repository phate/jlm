/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>

static int
test()
{
  using namespace jlm::llvm;

  /* setup cfg */

  ipgraph_module im(jlm::util::FilePath(""), "", "");

  jlm::llvm::cfg cfg(im);
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

  /* check orderings */

  std::vector<cfg_node *> po1({ cfg.exit(), bb3, bb2, bb1, bb0, cfg.entry() });
  std::vector<cfg_node *> po2({ cfg.exit(), bb3, bb1, bb2, bb0, cfg.entry() });
  assert(postorder(cfg) == po1 || postorder(cfg) == po2);

  std::vector<cfg_node *> rpo1({ cfg.entry(), bb0, bb1, bb2, bb3, cfg.exit() });
  std::vector<cfg_node *> rpo2({ cfg.entry(), bb0, bb2, bb1, bb3, cfg.exit() });
  assert(reverse_postorder(cfg) == rpo1 || reverse_postorder(cfg) == rpo2);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-cfg-orderings", test)
