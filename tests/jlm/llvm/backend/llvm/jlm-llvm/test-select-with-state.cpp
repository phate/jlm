/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/llvm/backend/IpGraphToLlvmConverter.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

static int
test()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();
  ipgraph_module m(jlm::util::filepath(""), "", "");

  std::unique_ptr<jlm::llvm::cfg> cfg(new jlm::llvm::cfg(m));
  auto bb = basic_block::create(*cfg);
  cfg->exit()->divert_inedges(bb);
  bb->add_outedge(cfg->exit());

  auto p = cfg->entry()->append_argument(argument::create("p", jlm::rvsdg::bittype::Create(1)));
  auto s1 = cfg->entry()->append_argument(argument::create("s1", mt));
  auto s2 = cfg->entry()->append_argument(argument::create("s2", mt));

  bb->append_last(SelectOperation::create(p, s1, s2));
  auto s3 = bb->last()->result(0);

  cfg->exit()->append_result(s3);
  cfg->exit()->append_result(s3);

  auto ft = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::bittype::Create(1), MemoryStateType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create(), MemoryStateType::Create() });
  auto f = function_node::create(m.ipgraph(), "f", ft, linkage::external_linkage);
  f->add_cfg(std::move(cfg));

  print(m, stdout);

  llvm::LLVMContext ctx;
  IpGraphToLlvmConverter::CreateAndConvertModule(m, ctx);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/jlm-llvm/test-select-with-state", test)
