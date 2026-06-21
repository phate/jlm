/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/Trace.hpp>

namespace jlm::llvm
{

GetElementPtrOperation::~GetElementPtrOperation() noexcept = default;

bool
GetElementPtrOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const GetElementPtrOperation *>(&other);

  if (operation == nullptr || getPointeeType() != operation->getPointeeType()
      || narguments() != operation->narguments())
  {
    return false;
  }

  for (size_t n = 0; n < narguments(); n++)
  {
    if (operation->argument(n) != argument(n))
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
}
