/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/types.hpp>
#include <jlm/util/Hash.hpp>
#include <jlm/util/strfmt.hpp>

#include <unordered_map>

namespace jlm::llvm
{

IntegerType::~IntegerType() noexcept = default;

std::string
IntegerType::debug_string() const
{
  return util::strfmt("i", NumBits());
}

bool
IntegerType::operator==(const Type & other) const noexcept
{
  const auto type = dynamic_cast<const IntegerType *>(&other);
  return type && type->NumBits() == NumBits();
}

std::size_t
IntegerType::ComputeHash() const noexcept
{
  const auto typeHash = typeid(IntegerType).hash_code();
  const auto numBitsHash = std::hash<size_t>()(NumBits());
  return util::CombineHashes(typeHash, numBitsHash);
}

std::shared_ptr<const IntegerType>
IntegerType::Create(std::size_t numBits)
{
  static const IntegerType static_instances[65] = {
    IntegerType(0),  IntegerType(1),  IntegerType(2),  IntegerType(3),  IntegerType(4),
    IntegerType(5),  IntegerType(6),  IntegerType(7),  IntegerType(8),  IntegerType(9),
    IntegerType(10), IntegerType(11), IntegerType(12), IntegerType(13), IntegerType(14),
    IntegerType(15), IntegerType(16), IntegerType(17), IntegerType(18), IntegerType(19),
    IntegerType(20), IntegerType(21), IntegerType(22), IntegerType(23), IntegerType(24),
    IntegerType(25), IntegerType(26), IntegerType(27), IntegerType(28), IntegerType(29),
    IntegerType(30), IntegerType(31), IntegerType(32), IntegerType(33), IntegerType(34),
    IntegerType(35), IntegerType(36), IntegerType(37), IntegerType(38), IntegerType(39),
    IntegerType(40), IntegerType(41), IntegerType(42), IntegerType(43), IntegerType(44),
    IntegerType(45), IntegerType(46), IntegerType(47), IntegerType(48), IntegerType(49),
    IntegerType(50), IntegerType(51), IntegerType(52), IntegerType(53), IntegerType(54),
    IntegerType(55), IntegerType(56), IntegerType(57), IntegerType(58), IntegerType(59),
    IntegerType(60), IntegerType(61), IntegerType(62), IntegerType(63), IntegerType(64)
  };

  if (numBits <= 64)
  {
    if (numBits == 0)
    {
      throw util::error("IntegerType::Create: Number of bits must be greater than zero.");
    }

    return std::shared_ptr<const IntegerType>(std::shared_ptr<void>(), &static_instances[numBits]);
  }

  return std::make_shared<IntegerType>(numBits);
}

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

}
