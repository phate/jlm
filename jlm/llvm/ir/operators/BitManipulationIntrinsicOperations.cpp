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

BSwapOperation::~BSwapOperation() noexcept = default;

bool
BSwapOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const BSwapOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
BSwapOperation::debug_string() const
{
  return util::strfmt("BSwap[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
BSwapOperation::copy() const
{
  return std::make_unique<BSwapOperation>(*this);
}

void
BSwapOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  auto scalarType = type.get();
  if (const auto vectorType = dynamic_cast<const VectorType *>(scalarType))
  {
    scalarType = &vectorType->type();
  }

  const auto bitType = dynamic_cast<const rvsdg::BitType *>(scalarType);
  if (!bitType)
  {
    throw std::runtime_error("BSwapOperation::checkType: Expected integer type.");
  }

  if (bitType->nbits() % 16 != 0)
  {
    throw std::runtime_error(
        "BSwapOperation::checkType: Expected integer type with a multiple of 16 bits.");
  }
}

CtlzOperation::~CtlzOperation() noexcept = default;

bool
CtlzOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const CtlzOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
CtlzOperation::debug_string() const
{
  return util::strfmt("Ctlz[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
CtlzOperation::copy() const
{
  return std::make_unique<CtlzOperation>(*this);
}

void
CtlzOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("CtlzOperation::checkType: Expected integer type.");
  }
}

CtpopOperation::~CtpopOperation() noexcept = default;

bool
CtpopOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const CtpopOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
CtpopOperation::debug_string() const
{
  return util::strfmt("Ctpop[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
CtpopOperation::copy() const
{
  return std::make_unique<CtpopOperation>(*this);
}

void
CtpopOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const rvsdg::BitType>(type) && !isVectorOf<const rvsdg::BitType>(*type))
  {
    throw std::runtime_error("CtpopOperation::checkType: Expected integer type.");
  }
}

}
