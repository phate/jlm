/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/ArithmeticWithOverflowIntrinsicOperations.hpp>

namespace jlm::llvm
{

SAddWithOverflowOperation::~SAddWithOverflowOperation() noexcept = default;

bool
SAddWithOverflowOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const SAddWithOverflowOperation *>(&other);
  return operation && *operation->getOperandType() == *getOperandType();
}

std::string
SAddWithOverflowOperation::debug_string() const
{
  return util::strfmt("SAddWithOverflow[", getOperandType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
SAddWithOverflowOperation::copy() const
{
  return std::make_unique<SAddWithOverflowOperation>(*this);
}

void
SAddWithOverflowOperation::checkOperandType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("SAddWithOverflowOperation::checkType: Expected integer type.");
  }
}

SSubWithOverflowOperation::~SSubWithOverflowOperation() noexcept = default;

bool
SSubWithOverflowOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const SSubWithOverflowOperation *>(&other);
  return operation && *operation->getOperandType() == *getOperandType();
}

std::string
SSubWithOverflowOperation::debug_string() const
{
  return util::strfmt("SSubWithOverflow[", getOperandType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
SSubWithOverflowOperation::copy() const
{
  return std::make_unique<SSubWithOverflowOperation>(*this);
}

void
SSubWithOverflowOperation::checkOperandType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("SSubWithOverflowOperation::checkType: Expected integer type.");
  }
}

SMulWithOverflowOperation::~SMulWithOverflowOperation() noexcept = default;

bool
SMulWithOverflowOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const SMulWithOverflowOperation *>(&other);
  return operation && *operation->getOperandType() == *getOperandType();
}

std::string
SMulWithOverflowOperation::debug_string() const
{
  return util::strfmt("SMulWithOverflow[", getOperandType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
SMulWithOverflowOperation::copy() const
{
  return std::make_unique<SMulWithOverflowOperation>(*this);
}

void
SMulWithOverflowOperation::checkOperandType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("SMulWithOverflowOperation::checkType: Expected integer type.");
  }
}

}
