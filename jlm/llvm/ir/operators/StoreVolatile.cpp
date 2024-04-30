/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/StoreVolatile.hpp>

namespace jlm::llvm
{

StoreVolatileOperation::~StoreVolatileOperation() noexcept = default;

bool
StoreVolatileOperation::operator==(const operation & other) const noexcept
{
  auto operation = dynamic_cast<const StoreVolatileOperation *>(&other);
  return operation && operation->NumMemoryStates() == NumMemoryStates()
      && operation->GetStoredType() == GetStoredType()
      && operation->GetAlignment() == GetAlignment();
}

std::string
StoreVolatileOperation::debug_string() const
{
  return "StoreVolatile";
}

std::unique_ptr<rvsdg::operation>
StoreVolatileOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new StoreVolatileOperation(*this));
}

rvsdg::node *
StoreVolatileNode::copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const
{
  return &CreateNode(*region, GetOperation(), operands);
}

}
