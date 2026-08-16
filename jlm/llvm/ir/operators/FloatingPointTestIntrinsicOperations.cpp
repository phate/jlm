/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/FloatingPointTestIntrinsicOperations.hpp>

namespace jlm::llvm
{

IsFPClassOperation::~IsFPClassOperation() noexcept = default;

bool
IsFPClassOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const IsFPClassOperation *>(&other);
  return operation && *operation->getType() == *getType();
}

std::string
IsFPClassOperation::debug_string() const
{
  return util::strfmt("IsFPClass[", getType()->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
IsFPClassOperation::copy() const
{
  return std::make_unique<IsFPClassOperation>(*this);
}

void
IsFPClassOperation::checkType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (!is<const FloatingPointType>(type) && !isVectorOf<const FloatingPointType>(*type))
  {
    throw std::runtime_error("IsFPClassOperation::checkType: Expected floating point type.");
  }
}

std::shared_ptr<const rvsdg::Type>
IsFPClassOperation::createResultType(const std::shared_ptr<const rvsdg::Type> & type)
{
  if (is<const FloatingPointType>(type))
    return rvsdg::BitType::Create(1);

  if (const auto fixedVectorType = std::dynamic_pointer_cast<const FixedVectorType>(type))
    return FixedVectorType::Create(rvsdg::BitType::Create(1), fixedVectorType->size());

  const auto scalableVectorType = std::dynamic_pointer_cast<const ScalableVectorType>(type);
  return ScalableVectorType::Create(rvsdg::BitType::Create(1), scalableVectorType->size());
}

}
