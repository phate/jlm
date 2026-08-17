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

PtrMaskOperation::~PtrMaskOperation() noexcept = default;

bool
PtrMaskOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const PtrMaskOperation *>(&other);
  return operation && *operation->getPtrOperandType() == *getPtrOperandType()
      && *operation->getMaskOperandType() == *getMaskOperandType();
}

std::string
PtrMaskOperation::debug_string() const
{
  return util::strfmt("PtrMask[", getMaskOperandType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
PtrMaskOperation::copy() const
{
  return std::make_unique<PtrMaskOperation>(*this);
}

void
PtrMaskOperation::checkOperandTypes(
    const std::shared_ptr<const rvsdg::Type> & ptrType,
    const std::shared_ptr<const rvsdg::Type> & maskType)
{
  if (is<PointerType>(ptrType))
  {
    if (is<rvsdg::BitType>(maskType))
      return;

    throw std::runtime_error(
        "PtrMaskOperation::checkOperandTypes: Expected mask type to be integer.");
  }

  if (const auto vectorType = std::dynamic_pointer_cast<const VectorType>(ptrType))
  {
    if (isVectorOfSize<const rvsdg::BitType>(*vectorType, vectorType->size()))
      return;

    throw std::runtime_error(util::strfmt(
        "PtrMaskOperation::checkOperandTypes: Expected mask type to be a "
        "vector of integers of size ",
        vectorType->size(),
        "."));
  }

  throw std::runtime_error(
      "PtrMaskOperation::checkOperandTypes: Expected pointer or vector of pointer types.");
}

}
