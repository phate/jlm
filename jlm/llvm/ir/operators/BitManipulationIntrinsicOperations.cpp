/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/BitManipulationIntrinsicOperations.hpp>

namespace jlm::llvm
{

FShlOperation::~FShlOperation() noexcept = default;

bool
FShlOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const FShlOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
FShlOperation::debug_string() const
{
  return util::strfmt("FShl[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
FShlOperation::copy() const
{
  return std::make_unique<FShlOperation>(*this);
}

void
FShlOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("FShlOperation::checkType: Expected integer type.");
  }
}

}
