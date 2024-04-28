/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/LoadVolatile.hpp>

namespace jlm::llvm
{

LoadVolatileOperation::~LoadVolatileOperation() noexcept = default;

bool
LoadVolatileOperation::operator==(const operation & other) const noexcept
{
  auto operation = dynamic_cast<const LoadVolatileOperation *>(&other);
  return operation && operation->narguments() == narguments()
      && operation->GetLoadedType() == GetLoadedType()
      && operation->GetAlignment() == GetAlignment();
}

std::string
LoadVolatileOperation::debug_string() const
{
  return "LoadVolatile";
}

std::unique_ptr<rvsdg::operation>
LoadVolatileOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new LoadVolatileOperation(*this));
}

rvsdg::node *
LoadVolatileNode::copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const
{
  return &CreateNode(*region, GetOperation(), operands);
}

}
