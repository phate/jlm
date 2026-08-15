/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/GeneralIntrinsicOperations.hpp>

namespace jlm::llvm
{

IsConstantOperation::~IsConstantOperation() noexcept = default;

bool
IsConstantOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const IsConstantOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
IsConstantOperation::debug_string() const
{
  return util::strfmt("IsConstant[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
IsConstantOperation::copy() const
{
  return std::make_unique<IsConstantOperation>(*this);
}

}
