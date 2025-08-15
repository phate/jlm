/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/util/Hash.hpp>
#include <jlm/util/Math.hpp>
#include <jlm/util/strfmt.hpp>

#include <numeric>
#include <unordered_map>

namespace jlm::llvm
{

PointerType::~PointerType() noexcept = default;

std::string
PointerType::debug_string() const
{
  return "ptr";
}

bool
PointerType::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  return jlm::rvsdg::is<PointerType>(other);
}

std::size_t
PointerType::ComputeHash() const noexcept
{
  return typeid(PointerType).hash_code();
}

std::shared_ptr<const PointerType>
PointerType::Create()
{
  static const PointerType instance;
  return std::shared_ptr<const PointerType>(std::shared_ptr<void>(), &instance);
}

ArrayType::~ArrayType() noexcept = default;

std::string
ArrayType::debug_string() const
{
  return util::strfmt("[ ", nelements(), " x ", type_->debug_string(), " ]");
}

bool
ArrayType::operator==(const Type & other) const noexcept
{
  const auto type = dynamic_cast<const ArrayType *>(&other);
  return type && type->element_type() == element_type() && type->nelements() == nelements();
}

std::size_t
ArrayType::ComputeHash() const noexcept
{
  const auto typeHash = typeid(ArrayType).hash_code();
  const auto numElementsHash = std::hash<std::size_t>()(nelements_);
  return util::CombineHashes(typeHash, type_->ComputeHash(), numElementsHash);
}

FloatingPointType::~FloatingPointType() noexcept = default;

std::string
FloatingPointType::debug_string() const
{
  static std::unordered_map<fpsize, std::string> map({ { fpsize::half, "half" },
                                                       { fpsize::flt, "float" },
                                                       { fpsize::dbl, "double" },
                                                       { fpsize::x86fp80, "x86fp80" },
                                                       { fpsize::fp128, "fp128" } });

  JLM_ASSERT(map.find(size()) != map.end());
  return map[size()];
}

bool
FloatingPointType::operator==(const Type & other) const noexcept
{
  const auto type = dynamic_cast<const FloatingPointType *>(&other);
  return type && type->size() == size();
}

std::size_t
FloatingPointType::ComputeHash() const noexcept
{
  const auto typeHash = typeid(FloatingPointType).hash_code();
  const auto sizeHash = std::hash<fpsize>()(size_);
  return util::CombineHashes(typeHash, sizeHash);
}

std::shared_ptr<const FloatingPointType>
FloatingPointType::Create(fpsize size)
{
  switch (size)
  {
  case fpsize::half:
  {
    static const FloatingPointType instance(fpsize::half);
    return std::shared_ptr<const FloatingPointType>(std::shared_ptr<void>(), &instance);
  }
  case fpsize::flt:
  {
    static const FloatingPointType instance(fpsize::flt);
    return std::shared_ptr<const FloatingPointType>(std::shared_ptr<void>(), &instance);
  }
  case fpsize::dbl:
  {
    static const FloatingPointType instance(fpsize::dbl);
    return std::shared_ptr<const FloatingPointType>(std::shared_ptr<void>(), &instance);
  }
  case fpsize::x86fp80:
  {
    static const FloatingPointType instance(fpsize::x86fp80);
    return std::shared_ptr<const FloatingPointType>(std::shared_ptr<void>(), &instance);
  }
  case fpsize::fp128:
  {
    static const FloatingPointType instance(fpsize::fp128);
    return std::shared_ptr<const FloatingPointType>(std::shared_ptr<void>(), &instance);
  }
  default:
  {
    JLM_UNREACHABLE("unknown fpsize");
  }
  }
}

VariableArgumentType::~VariableArgumentType() noexcept = default;

bool
VariableArgumentType::operator==(const Type & other) const noexcept
{
  return dynamic_cast<const VariableArgumentType *>(&other) != nullptr;
}

std::size_t
VariableArgumentType::ComputeHash() const noexcept
{
  return typeid(VariableArgumentType).hash_code();
}

std::string
VariableArgumentType::debug_string() const
{
  return "vararg";
}

std::shared_ptr<const VariableArgumentType>
VariableArgumentType::Create()
{
  static const VariableArgumentType instance;
  return std::shared_ptr<const VariableArgumentType>(std::shared_ptr<void>(), &instance);
}

StructType::~StructType() = default;

bool
StructType::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  auto type = dynamic_cast<const StructType *>(&other);
  return type && type->IsPacked_ == IsPacked_ && type->Name_ == Name_
      && &type->Declaration_ == &Declaration_;
}

std::size_t
StructType::ComputeHash() const noexcept
{
  auto typeHash = typeid(StructType).hash_code();
  auto isPackedHash = std::hash<bool>()(IsPacked_);
  auto nameHash = std::hash<std::string>()(Name_);
  auto declarationHash = std::hash<const StructType::Declaration *>()(&Declaration_);
  return util::CombineHashes(typeHash, isPackedHash, nameHash, declarationHash);
}

std::string
StructType::debug_string() const
{
  return "struct";
}

size_t
StructType::GetFieldOffset(size_t fieldIndex) const
{
  const auto & decl = GetDeclaration();
  const auto isPacked = IsPacked();

  size_t offset = 0;

  for (size_t i = 0; i < decl.NumElements(); i++)
  {
    auto & field = decl.GetElement(i);

    // First round up to the alignment of the field
    auto fieldAlignment = isPacked ? 1 : GetTypeAlignment(field);
    offset = util::RoundUpToMultipleOf(offset, fieldAlignment);

    if (i == fieldIndex)
      return offset;

    // Add the size of the field
    offset += GetTypeSize(field);
  }

  JLM_UNREACHABLE("Invalid fieldIndex in GetStructFieldOffset");
}

bool
VectorType::operator==(const rvsdg::Type & other) const noexcept
{
  const auto type = dynamic_cast<const VectorType *>(&other);
  return type && type->size_ == size_ && *type->type_ == *type_;
}

FixedVectorType::~FixedVectorType() noexcept = default;

bool
FixedVectorType::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  return VectorType::operator==(other);
}

std::size_t
FixedVectorType::ComputeHash() const noexcept
{
  auto typeHash = typeid(FixedVectorType).hash_code();
  auto sizeHash = std::hash<size_t>()(size());
  return util::CombineHashes(typeHash, sizeHash, Type()->ComputeHash());
}

std::string
FixedVectorType::debug_string() const
{
  return util::strfmt("fixedvector[", type().debug_string(), ":", size(), "]");
}

ScalableVectorType::~ScalableVectorType() noexcept = default;

bool
ScalableVectorType::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  return VectorType::operator==(other);
}

std::size_t
ScalableVectorType::ComputeHash() const noexcept
{
  const auto typeHash = typeid(ScalableVectorType).hash_code();
  const auto sizeHash = std::hash<size_t>()(size());
  return util::CombineHashes(typeHash, sizeHash, Type()->ComputeHash());
}

std::string
ScalableVectorType::debug_string() const
{
  return util::strfmt("scalablevector[", type().debug_string(), ":", size(), "]");
}

IOStateType::~IOStateType() noexcept = default;

bool
IOStateType::operator==(const Type & other) const noexcept
{
  return jlm::rvsdg::is<IOStateType>(other);
}

std::size_t
IOStateType::ComputeHash() const noexcept
{
  return typeid(IOStateType).hash_code();
}

std::string
IOStateType::debug_string() const
{
  return "iostate";
}

std::shared_ptr<const IOStateType>
IOStateType::Create()
{
  static const IOStateType instance;
  return std::shared_ptr<const IOStateType>(std::shared_ptr<void>(), &instance);
}

/**
 * MemoryStateType class
 */
MemoryStateType::~MemoryStateType() noexcept = default;

std::string
MemoryStateType::debug_string() const
{
  return "mem";
}

bool
MemoryStateType::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  return jlm::rvsdg::is<MemoryStateType>(other);
}

std::size_t
MemoryStateType::ComputeHash() const noexcept
{
  return typeid(MemoryStateType).hash_code();
}

std::shared_ptr<const MemoryStateType>
MemoryStateType::Create()
{
  static const MemoryStateType instance;
  return std::shared_ptr<const MemoryStateType>(std::shared_ptr<void>(), &instance);
}

size_t
GetTypeSize(const rvsdg::ValueType & type)
{
  if (const auto bits = dynamic_cast<const rvsdg::BitType *>(&type))
  {
    // Assume 8 bits per byte, and round up to a power of 2 bytes
    const auto bytes = (bits->nbits() + 7) / 8;
    return util::RoundUpToPowerOf2(bytes);
  }
  if (jlm::rvsdg::is<PointerType>(type))
  {
    // FIXME: Use the target information in the module to find the actual size of pointers
    return 8;
  }
  if (const auto arrayType = dynamic_cast<const ArrayType *>(&type))
  {
    return arrayType->nelements() * GetTypeSize(*arrayType->GetElementType());
  }
  if (const auto floatType = dynamic_cast<const FloatingPointType *>(&type))
  {
    switch (floatType->size())
    {
    case fpsize::half:
      return 2;
    case fpsize::flt:
      return 4;
    case fpsize::dbl:
      return 8;
    case fpsize::fp128:
      return 16;
    case fpsize::x86fp80:
      return 16; // Will never actually be written to memory, but we round up
    default:
      JLM_UNREACHABLE("Unknown float size");
    }
  }
  if (const auto structType = dynamic_cast<const StructType *>(&type))
  {
    size_t totalSize = 0;
    size_t alignment = 1;

    const auto & decl = structType->GetDeclaration();
    // A packed struct has alignment 1, and all fields are tightly packed
    const auto isPacked = structType->IsPacked();

    for (size_t i = 0; i < decl.NumElements(); i++)
    {
      auto & field = decl.GetElement(i);
      auto fieldSize = GetTypeSize(field);
      auto fieldAlignment = isPacked ? 1 : GetTypeAlignment(field);

      // Add the size of the field, including any needed padding
      totalSize = util::RoundUpToMultipleOf(totalSize, fieldAlignment);
      totalSize += fieldSize;

      // The struct as a whole must be at least as aligned as each field
      alignment = std::lcm(alignment, fieldAlignment);
    }

    // Round size up to a multiple of alignment
    totalSize = util::RoundUpToMultipleOf(totalSize, alignment);

    // If the struct has 0 fields, its size becomes 0.
    // In C++, where types of size 0 are forbidden, clang will have inserted a dummy i8 field.

    return totalSize;
  }
  if (const auto vectorType = dynamic_cast<const VectorType *>(&type))
  {
    // In LLVM, vectors always have alignment >= the number of bytes of data stored in the vector
    const auto bytesNeeded = vectorType->size() * GetTypeSize(*vectorType->Type());
    return util::RoundUpToPowerOf2(bytesNeeded);
  }
  if (jlm::rvsdg::is<rvsdg::FunctionType>(type))
  {
    // Functions should never read from or written to, so give them size 0
    // Note: this is not the same as a function pointer, which is a PointerType
    return 0;
  }

  JLM_UNREACHABLE(util::strfmt("Unknown type: ", typeid(type).name()).c_str());
}

size_t
GetTypeAlignment(const rvsdg::ValueType & type)
{
  if (jlm::rvsdg::is<rvsdg::BitType>(type) || jlm::rvsdg::is<PointerType>(type)
      || jlm::rvsdg::is<FloatingPointType>(type) || jlm::rvsdg::is<VectorType>(type))
  {
    // These types all have alignment equal to their size
    return GetTypeSize(type);
  }
  if (const auto arrayType = dynamic_cast<const ArrayType *>(&type))
  {
    return GetTypeAlignment(*arrayType->GetElementType());
  }
  if (const auto structType = dynamic_cast<const StructType *>(&type))
  {
    const auto & decl = structType->GetDeclaration();
    // A packed struct has alignment 1, and all fields are tightly packed
    if (structType->IsPacked())
      return 1;

    size_t alignment = 1;

    for (size_t i = 0; i < decl.NumElements(); i++)
    {
      auto & field = decl.GetElement(i);
      auto fieldAlignment = GetTypeAlignment(field);

      // The struct as a whole must be at least as aligned as each field
      alignment = std::lcm(alignment, fieldAlignment);
    }

    return alignment;
  }

  JLM_UNREACHABLE("Unknown type");
}

}
