/*
 * Copyright 2025 Felix Reißmann <felix.rm.153@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

static int
test()
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  ipgraph_module m(jlm::util::filepath(""), "", "");

  std::unique_ptr<jlm::llvm::cfg> cfg(new jlm::llvm::cfg(m));
  auto bb = basic_block::create(*cfg);
  cfg->exit()->divert_inedges(bb);
  bb->add_outedge(cfg->exit());

  bb->append_last(UndefValueOperation::Create(mt, "s1"));
  auto s1 = bb->last()->result(0);

  cfg->exit()->append_result(s1);

  auto ft = FunctionType::Create({}, { mt });
  auto f = function_node::create(m.ipgraph(), "f", ft, linkage::external_linkage);
  f->add_cfg(std::move(cfg));

  llvm::LLVMContext ctx;
  jlm2llvm::convert(m, ctx);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/jlm-llvm/test-ignore-memory-state", test)
