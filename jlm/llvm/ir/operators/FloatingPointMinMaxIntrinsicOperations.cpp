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

RoundOperation::~RoundOperation() noexcept = default;

bool
RoundOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const RoundOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
RoundOperation::debug_string() const
{
  return util::strfmt("Round[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
RoundOperation::copy() const
{
  return std::make_unique<RoundOperation>(*this);
}

void
RoundOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const FloatingPointType>(type) && !isVectorOf<const FloatingPointType>(*type))
  {
    throw std::runtime_error("RoundOperation::checkType: Expected floating point type.");
  }
}

TruncIntrinsicOperation::~TruncIntrinsicOperation() noexcept = default;

bool
TruncIntrinsicOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const TruncIntrinsicOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
TruncIntrinsicOperation::debug_string() const
{
  return util::strfmt("TruncIntrinsic[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
TruncIntrinsicOperation::copy() const
{
  return std::make_unique<TruncIntrinsicOperation>(*this);
}

void
TruncIntrinsicOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const FloatingPointType>(type) && !isVectorOf<const FloatingPointType>(*type))
  {
    throw std::runtime_error("TruncIntrinsicOperation::checkType: Expected floating point type.");
  }
}

CopysignOperation::~CopysignOperation() noexcept = default;

bool
CopysignOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const CopysignOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
CopysignOperation::debug_string() const
{
  return util::strfmt("Copysign[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
CopysignOperation::copy() const
{
  return std::make_unique<CopysignOperation>(*this);
}

void
CopysignOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const FloatingPointType>(type) && !isVectorOf<const FloatingPointType>(*type))
  {
    throw std::runtime_error("CopysignOperation::checkType: Expected floating point type.");
  }
}

}
