/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "IntegerOperations.hpp"
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/Trace.hpp>

namespace jlm::llvm
{

GetElementPtrOperation::~GetElementPtrOperation() noexcept = default;

bool
GetElementPtrOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const GetElementPtrOperation *>(&other);

  if (operation == nullptr || *getPointeeType() != *operation->getPointeeType()
      || narguments() != operation->narguments())
  {
    return false;
  }

  for (size_t n = 0; n < narguments(); n++)
  {
    if (*operation->argument(n) != *argument(n))
    {
      return false;
    }
  }

  return true;
}

std::string
GetElementPtrOperation::debug_string() const
{
  return "GetElementPtr";
}

std::unique_ptr<rvsdg::Operation>
GetElementPtrOperation::copy() const
{
  return std::make_unique<GetElementPtrOperation>(*this);
}

std::optional<GetElementPtrOperation::Constant>
GetElementPtrOperation::tryGetAsConstant(const rvsdg::SimpleNode & gepNode)
{
  const auto gepOperation = dynamic_cast<const GetElementPtrOperation *>(&gepNode.GetOperation());
  if (!gepOperation)
    return std::nullopt;

  std::vector<uint64_t> indices;
  for (auto & input : gepOperation->indices(gepNode))
  {
    if (auto indexOpt = tryGetConstantSignedInteger(*input.origin()))
    {
      indices.push_back(indexOpt.value());
    }
    else
    {
      return std::nullopt;
    }
  }

  return Constant{ gepOperation->getPointeeType(), indices };
}

int64_t
GetElementPtrOperation::Constant::getOffsetInBytes() const noexcept
{
  JLM_ASSERT(indices.size() >= 1);

  std::function<uint64_t(size_t, const rvsdg::Type &)> computeIntraTypeOffset =
      [&](const size_t index, const rvsdg::Type & type)
  {
    if (index >= indices.size())
      return static_cast<int64_t>(0);

    const auto indexValue = indices[index];
    if (const auto arrayType = dynamic_cast<const ArrayType *>(&type))
    {
      const auto & elementType = *arrayType->GetElementType();
      int64_t offsetInBytes = indexValue * GetTypeAllocSize(elementType);
      offsetInBytes += computeIntraTypeOffset(index + 1, elementType);
      return offsetInBytes;
    }

    if (const auto structType = dynamic_cast<const StructType *>(&type))
    {
      const auto & fieldType = *structType->getElementType(indexValue);
      int64_t offsetInBytes = structType->GetFieldOffset(indexValue);
      offsetInBytes += computeIntraTypeOffset(index + 1, fieldType);
      return offsetInBytes;
    }

    throw std::logic_error("Unknown GetElementPtr type");
  };

  const auto wholeTypeIndex = indices[0];
  int64_t offsetInBytes = wholeTypeIndex * GetTypeAllocSize(*pointeeType);
  offsetInBytes += computeIntraTypeOffset(1, *pointeeType);
  return offsetInBytes;
}

std::shared_ptr<const rvsdg::Type>
GetElementPtrOperation::getIndexedType(
    const std::shared_ptr<const rvsdg::Type> & gepType,
    const std::vector<rvsdg::Output *> & indices)
{
  if (indices.empty())
    return gepType;

  auto currentType = gepType;

  // We skip the first index as it always just steps through the container
  for (size_t n = 1; n < indices.size(); ++n)
  {
    if (auto structType = std::dynamic_pointer_cast<const StructType>(currentType))
    {
      auto index = indices[n];
      auto & tracedIndex = llvm::traceOutput(*index);
      auto [constantNode, constantOperation] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(tracedIndex);
      if (!constantOperation)
      {
        return nullptr;
      }

      if (constantOperation->Representation().nbits() != 32)
        return nullptr;

      auto idx = constantOperation->Representation().to_uint();
      if (idx > structType->numElements())
        return nullptr;

      currentType = structType->getElementType(idx);
    }
    else if (const auto arrayType = std::dynamic_pointer_cast<const ArrayType>(currentType))
    {
      currentType = arrayType->GetElementType();
    }
    else if (const auto vectorType = std::dynamic_pointer_cast<const VectorType>(currentType))
    {
      currentType = vectorType->Type();
    }
    else
    {
      return nullptr;
    }
  }

  return currentType;
}

}
