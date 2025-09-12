/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/frontend/ControlFlowRestructuring.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>

static void
AcyclicStructured()
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

  assert(nnodes == cfg.nnodes());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/test-restructuring-AcyclicStructured",
    AcyclicStructured)

static void
AcyclicUnstructured()
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

  assert(is_proper_structured(cfg));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/test-restructuring-AcyclicUnstructured",
    AcyclicUnstructured)

static void
DoWhileLoop()
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

  assert(nnodes == cfg.nnodes());
  assert(bb2->OutEdge(0)->sink() == bb2);
  assert(bb3->OutEdge(0)->sink() == bb1);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-restructuring-DoWhileLoop", DoWhileLoop)

static void
WhileLoop()
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

  assert(is_proper_structured(cfg));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-restructuring-WhileLoop", WhileLoop)

static void
IrreducibleCfg()
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
  assert(is_proper_structured(cfg));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-restructuring-IrreducibleCfg", IrreducibleCfg)

static void
AcyclicUnstructuredInDoWhileLoop()
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
  assert(is_proper_structured(cfg));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/test-restructuring-AcyclicUnstructuredInDoWhileLoop",
    AcyclicUnstructuredInDoWhileLoop)

static void
LorBeforeDoWhileLoop()
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
  assert(is_proper_structured(cfg));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/test-restructuring-LorBeforeDoWhileLoop",
    LorBeforeDoWhileLoop)

static void
StaticEndlessLoop()
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
  assert(is_proper_structured(cfg));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/test-restructuring-StaticEndlessLoop",
    StaticEndlessLoop)
