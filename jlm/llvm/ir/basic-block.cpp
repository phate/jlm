/*
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <sstream>

namespace jlm::llvm
{

BasicBlock::~BasicBlock() noexcept = default;

bool
BasicBlock::HasSsaPhiOperation() const
{
  return is<SsaPhiOperation>(first());
}

llvm::ThreeAddressCode *
BasicBlock::insert_before_branch(std::unique_ptr<llvm::ThreeAddressCode> tac)
{
  auto it = is<BranchOperation>(last()) ? std::prev(end()) : end();
  return insert_before(it, std::move(tac));
}

void
BasicBlock::insert_before_branch(tacsvector_t & tv)
{
  auto it = is<BranchOperation>(last()) ? std::prev(end()) : end();
  insert_before(it, tv);
}

BasicBlock *
BasicBlock::create(ControlFlowGraph & cfg)
{
  std::unique_ptr<BasicBlock> node(new BasicBlock(cfg));
  return static_cast<BasicBlock *>(cfg.add_node(std::move(node)));
}

}
