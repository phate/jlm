/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/SpecializedArithmeticIntrinsicOperations.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::llvm
{

FMulAddIntrinsicOperation::~FMulAddIntrinsicOperation() noexcept = default;

bool
FMulAddIntrinsicOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const FMulAddIntrinsicOperation *>(&other);
  return operation && operation->result(0) == result(0);
}

std::string
FMulAddIntrinsicOperation::debug_string() const
{
  return util::strfmt("FMulAddIntrinsic[", result(0)->debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
FMulAddIntrinsicOperation::copy() const
{
  return std::make_unique<FMulAddIntrinsicOperation>(*this);
}

void
FMulAddIntrinsicOperation::CheckType(const std::shared_ptr<const rvsdg::Type> & type)
{
  std::shared_ptr<const rvsdg::Type> scalarType = type;

  if (const auto vectorType = std::dynamic_pointer_cast<const VectorType>(type))
  {
    scalarType = std::static_pointer_cast<const rvsdg::Type>(vectorType->Type());
  }

  const auto fpType = std::dynamic_pointer_cast<const FloatingPointType>(scalarType);
  if (!fpType)
  {
    throw std::runtime_error(
        "FMulAddIntrinsicOperation::CheckAndExtractType: Expected floating point type.");
  }
}

}
