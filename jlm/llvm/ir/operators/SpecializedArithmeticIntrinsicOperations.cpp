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

std::shared_ptr<const FloatingPointType>
FMulAddIntrinsicOperation::CheckAndExtractType(const std::shared_ptr<const rvsdg::Type> & type)
{
  const auto fpType = std::dynamic_pointer_cast<const FloatingPointType>(type);
  if (!fpType)
  {
    throw std::runtime_error(
        "FMulAddIntrinsicOperation::CheckAndExtractType: Expected floating point type.");
  }

  if (fpType->size() != fpsize::flt && fpType->size() != fpsize::dbl)
  {
    throw std::runtime_error("FMulAddIntrinsicOperation::CheckAndExtractType: Expected float or "
                             "double floating point type size.");
  }

  return fpType;
}

}
