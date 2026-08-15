/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/FloatingPointMinMaxIntrinsicOperations.hpp>

namespace jlm::llvm
{

FloorOperation::~FloorOperation() noexcept = default;

bool
FloorOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const FloorOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
FloorOperation::debug_string() const
{
  return util::strfmt("Floor[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
FloorOperation::copy() const
{
  return std::make_unique<FloorOperation>(*this);
}

void
FloorOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const FloatingPointType>(type) && !isVectorOf<const FloatingPointType>(*type))
  {
    throw std::runtime_error("FloorOperation::checkType: Expected floating point type.");
  }
}

CeilOperation::~CeilOperation() noexcept = default;

bool
CeilOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const CeilOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
CeilOperation::debug_string() const
{
  return util::strfmt("Ceil[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
CeilOperation::copy() const
{
  return std::make_unique<CeilOperation>(*this);
}

void
CeilOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const FloatingPointType>(type) && !isVectorOf<const FloatingPointType>(*type))
  {
    throw std::runtime_error("CeilOperation::checkType: Expected floating point type.");
  }
}

}
