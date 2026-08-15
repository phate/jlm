/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>

namespace jlm::llvm
{

MemCpyNonVolatileOperation::~MemCpyNonVolatileOperation() = default;

bool
MemCpyNonVolatileOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemCpyNonVolatileOperation *>(&other);
  return operation && operation->LengthType() == LengthType()
      && operation->NumMemoryStates() == NumMemoryStates();
}

std::string
MemCpyNonVolatileOperation::debug_string() const
{
  return "MemCpy";
}

std::unique_ptr<rvsdg::Operation>
MemCpyNonVolatileOperation::copy() const
{
  return std::make_unique<MemCpyNonVolatileOperation>(*this);
}

size_t
MemCpyNonVolatileOperation::NumMemoryStates() const noexcept
{
  return nresults();
}

MemCpyVolatileOperation::~MemCpyVolatileOperation() noexcept = default;

bool
MemCpyVolatileOperation::operator==(const Operation & other) const noexcept
{
  // Avoid common node elimination for memcpy operator
  return this == &other;
}

std::string
MemCpyVolatileOperation::debug_string() const
{
  return "MemCpyVolatile";
}

std::unique_ptr<rvsdg::Operation>
MemCpyVolatileOperation::copy() const
{
  return std::make_unique<MemCpyVolatileOperation>(*this);
}

size_t
MemCpyVolatileOperation::NumMemoryStates() const noexcept
{
  // Subtracting I/O state
  return nresults() - 1;
}

MemSetNonVolatileOperation::~MemSetNonVolatileOperation() noexcept = default;

bool
MemSetNonVolatileOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const MemSetNonVolatileOperation *>(&other);
  return operation && operation->lengthType() == lengthType()
      && operation->numMemoryStates() == numMemoryStates();
}

std::string
MemSetNonVolatileOperation::debug_string() const
{
  return "MemSet";
}

std::unique_ptr<rvsdg::Operation>
MemSetNonVolatileOperation::copy() const
{
  return std::make_unique<MemSetNonVolatileOperation>(*this);
}

size_t
MemSetNonVolatileOperation::numMemoryStates() const noexcept
{
  return nresults();
}

UMaxOperation::~UMaxOperation() noexcept = default;

bool
UMaxOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const UMaxOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
UMaxOperation::debug_string() const
{
  return util::strfmt("UMax[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
UMaxOperation::copy() const
{
  return std::make_unique<UMaxOperation>(*this);
}

void
UMaxOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("UMaxOperation::checkType: Expected integer type.");
  }
}

SMinOperation::~SMinOperation() noexcept = default;

bool
SMinOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const SMinOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
SMinOperation::debug_string() const
{
  return util::strfmt("SMin[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
SMinOperation::copy() const
{
  return std::make_unique<SMinOperation>(*this);
}

void
SMinOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("SMinOperation::checkType: Expected integer type.");
  }
}

UMinOperation::~UMinOperation() noexcept = default;

bool
UMinOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const UMinOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
UMinOperation::debug_string() const
{
  return util::strfmt("UMin[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
UMinOperation::copy() const
{
  return std::make_unique<UMinOperation>(*this);
}

void
UMinOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("UMinOperation::checkType: Expected integer type.");
  }
}

}
