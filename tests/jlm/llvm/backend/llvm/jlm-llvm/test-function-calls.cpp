/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */
#include "test-registry.hpp"
#include "test-util.hpp"

#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

static void
test_malloc()
{
  auto setup = []()
  {
    using namespace jlm::llvm;

    auto mt = MemoryStateType::Create();
    auto pt = PointerType::Create();
    auto im = ipgraph_module::create(jlm::util::filepath(""), "", "");

    auto cfg = cfg::create(*im);
    auto bb = basic_block::create(*cfg);
    cfg->exit()->divert_inedges(bb);
    bb->add_outedge(cfg->exit());

    auto size =
        cfg->entry()->append_argument(argument::create("size", jlm::rvsdg::bittype::Create(64)));

    bb->append_last(malloc_op::create(size));

    cfg->exit()->append_result(bb->last()->result(0));
    cfg->exit()->append_result(bb->last()->result(1));

    auto ft = jlm::rvsdg::FunctionType::Create(
        { jlm::rvsdg::bittype::Create(64) },
        { PointerType::Create(), MemoryStateType::Create() });
    auto f = function_node::create(im->ipgraph(), "f", ft, linkage::external_linkage);
    f->add_cfg(std::move(cfg));

    return im;
  };

  auto verify = [](const llvm::Module & m)
  {
    using namespace llvm;

    auto f = m.getFunction("f");
    auto & bb = f->getEntryBlock();

    assert(bb.sizeWithoutDebug() == 2);
    assert(bb.getFirstNonPHI()->getOpcode() == llvm::Instruction::Call);
    assert(bb.getTerminator()->getOpcode() == llvm::Instruction::Ret);
  };

  auto im = setup();
  print(*im, stdout);

  llvm::LLVMContext ctx;
  auto lm = jlm::llvm::jlm2llvm::convert(*im, ctx);
  jlm::tests::print(*lm);

  verify(*lm);
}

static void
test_free()
{
  auto setup = []()
  {
    using namespace jlm::llvm;

    auto iot = IOStateType::Create();
    auto mt = MemoryStateType::Create();
    auto pt = PointerType::Create();

    auto ipgmod = ipgraph_module::create(jlm::util::filepath(""), "", "");

    auto ft = jlm::rvsdg::FunctionType::Create(
        { PointerType::Create(), MemoryStateType::Create(), IOStateType::Create() },
        { MemoryStateType::Create(), IOStateType::Create() });
    auto f = function_node::create(ipgmod->ipgraph(), "f", ft, linkage::external_linkage);

    auto cfg = cfg::create(*ipgmod);
    auto arg0 = cfg->entry()->append_argument(argument::create("pointer", pt));
    auto arg1 = cfg->entry()->append_argument(argument::create("memstate", mt));
    auto arg2 = cfg->entry()->append_argument(argument::create("iostate", iot));

    auto bb = basic_block::create(*cfg);
    cfg->exit()->divert_inedges(bb);
    bb->add_outedge(cfg->exit());

    bb->append_last(FreeOperation::Create(arg0, { arg1 }, arg2));

    cfg->exit()->append_result(bb->last()->result(0));
    cfg->exit()->append_result(bb->last()->result(1));

    f->add_cfg(std::move(cfg));

    return ipgmod;
  };

  auto verify = [](const llvm::Module & module)
  {
    using namespace llvm;

    auto f = module.getFunction("f");
    auto & bb = f->getEntryBlock();

    assert(bb.sizeWithoutDebug() == 2);
    assert(bb.getFirstNonPHI()->getOpcode() == Instruction::Call);
    assert(bb.getTerminator()->getOpcode() == Instruction::Ret);
  };

  auto ipgmod = setup();
  print(*ipgmod, stdout);

  llvm::LLVMContext ctx;
  auto llvmmod = jlm::llvm::jlm2llvm::convert(*ipgmod, ctx);
  jlm::tests::print(*llvmmod);

  verify(*llvmmod);
}

static int
test()
{
  test_malloc();
  test_free();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/jlm-llvm/test-function-calls", test)
