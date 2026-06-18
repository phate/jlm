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

/**
 * Calculates the byte offset inside the given type, starting at the given offset of GEP inputs.
 * Uses recursion to handle nested types.
 * If any indexing input is not a compile time constant, nullopt is returned.
 * @param gepNode the GEP node
 * @param inputIndex the index of the input that applies inside the given type
 * @param type the type the offset is inside
 * @return the byte offset within the given type, or nullopt if not possible.
 */
static std::optional<int64_t>
CalculateIntraTypeGepOffset(
    const rvsdg::SimpleNode & gepNode,
    const size_t inputIndex,
    const rvsdg::Type & type)
{
  // If we have no more input index values, we are not offsetting into the type
  if (inputIndex >= gepNode.ninputs())
    return 0;

  // GEP input 0 is the pointer being offset
  // GEP input 1 is the number of whole types
  // Intra-type offsets start at input 2 and beyond
  JLM_ASSERT(inputIndex >= 2);

  auto & gepInput = *gepNode.input(inputIndex)->origin();
  auto indexingValue = tryGetConstantSignedInteger(gepInput);

  // Any unknown indexing value means the GEP offset is unknown overall
  if (!indexingValue.has_value())
    return std::nullopt;

  if (auto array = dynamic_cast<const ArrayType *>(&type))
  {
    const auto & elementType = array->GetElementType();
    int64_t offset = *indexingValue * GetTypeAllocSize(*elementType);

    // Get the offset into the element type as well, if any
    const auto subOffset = CalculateIntraTypeGepOffset(gepNode, inputIndex + 1, *elementType);
    if (subOffset.has_value())
      return offset + *subOffset;

    return std::nullopt;
  }
  if (auto strct = dynamic_cast<const StructType *>(&type))
  {
    if (*indexingValue < 0 || static_cast<size_t>(*indexingValue) >= strct->numElements())
      throw std::logic_error("Struct type has fewer fields than requested by GEP");

    const auto & fieldType = strct->getElementType(*indexingValue);
    int64_t offset = strct->GetFieldOffset(*indexingValue);

    const auto subOffset = CalculateIntraTypeGepOffset(gepNode, inputIndex + 1, *fieldType);
    if (subOffset.has_value())
      return offset + *subOffset;

    return std::nullopt;
  }

  JLM_UNREACHABLE("Unknown GEP type");
}

std::optional<int64_t>
GetElementPtrOperation::CalculateOffset(const rvsdg::SimpleNode & gepNode)
{
  if (!is<GetElementPtrOperation>(gepNode.GetOperation()))
  {
    return std::nullopt;
  }
  const auto gep = static_cast<const GetElementPtrOperation *>(&gepNode.GetOperation());

  // The pointee type. Gets updated by the loop below if the GEP has multiple levels of offsets
  const auto & pointeeType = gep->getPointeeType();

  const auto & wholeTypeIndexingOrigin = *gepNode.input(1)->origin();
  const auto wholeTypeIndexing = tryGetConstantSignedInteger(wholeTypeIndexingOrigin);

  if (!wholeTypeIndexing.has_value())
    return std::nullopt;

  const int64_t offset = *wholeTypeIndexing * GetTypeAllocSize(*pointeeType);

  // In addition to offsetting by whole types, a GEP can also offset within a type
  const auto subOffset = CalculateIntraTypeGepOffset(gepNode, 2, *pointeeType);
  if (!subOffset.has_value())
    return std::nullopt;

  return offset + *subOffset;
}

int64_t
GetElementPtrOperation::Constant::getOffsetInBytes() const noexcept
{
  JLM_ASSERT(indices.size() >= 2);

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
