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

/* array type */

arraytype::~arraytype()
{}

std::string
arraytype::debug_string() const
{
  return util::strfmt("[ ", nelements(), " x ", type_->debug_string(), " ]");
}

bool
arraytype::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  auto type = dynamic_cast<const arraytype *>(&other);
  return type && type->element_type() == element_type() && type->nelements() == nelements();
}

std::size_t
arraytype::ComputeHash() const noexcept
{
  auto typeHash = typeid(arraytype).hash_code();
  auto numElementsHash = std::hash<std::size_t>()(nelements_);
  return util::CombineHashes(typeHash, type_->ComputeHash(), numElementsHash);
}

/* floating point type */

fptype::~fptype()
{}

std::string
fptype::debug_string() const
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
fptype::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  auto type = dynamic_cast<const fptype *>(&other);
  return type && type->size() == size();
}

std::size_t
fptype::ComputeHash() const noexcept
{
  auto typeHash = typeid(fptype).hash_code();
  auto sizeHash = std::hash<fpsize>()(size_);

  return util::CombineHashes(typeHash, sizeHash);
}

std::shared_ptr<const fptype>
fptype::Create(fpsize size)
{
  switch (size)
  {
  case fpsize::half:
  {
    static const fptype instance(fpsize::half);
    return std::shared_ptr<const fptype>(std::shared_ptr<void>(), &instance);
  }
  case fpsize::flt:
  {
    static const fptype instance(fpsize::flt);
    return std::shared_ptr<const fptype>(std::shared_ptr<void>(), &instance);
  }
  case fpsize::dbl:
  {
    static const fptype instance(fpsize::dbl);
    return std::shared_ptr<const fptype>(std::shared_ptr<void>(), &instance);
  }
  case fpsize::x86fp80:
  {
    static const fptype instance(fpsize::x86fp80);
    return std::shared_ptr<const fptype>(std::shared_ptr<void>(), &instance);
  }
  case fpsize::fp128:
  {
    static const fptype instance(fpsize::fp128);
    return std::shared_ptr<const fptype>(std::shared_ptr<void>(), &instance);
  }
  default:
  {
    JLM_UNREACHABLE("unknown fpsize");
  }
  }
}

/* vararg type */

varargtype::~varargtype()
{}

bool
varargtype::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  return dynamic_cast<const varargtype *>(&other) != nullptr;
}

std::size_t
varargtype::ComputeHash() const noexcept
{
  return typeid(varargtype).hash_code();
}

std::string
varargtype::debug_string() const
{
  return "vararg";
}

std::shared_ptr<const varargtype>
varargtype::Create()
{
  static const varargtype instance;
  return std::shared_ptr<const varargtype>(std::shared_ptr<void>(), &instance);
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

/* vectortype */

bool
vectortype::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  auto type = dynamic_cast<const vectortype *>(&other);
  return type && type->size_ == size_ && *type->type_ == *type_;
}

/* fixedvectortype */

fixedvectortype::~fixedvectortype()
{}

bool
fixedvectortype::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  return vectortype::operator==(other);
}

std::size_t
fixedvectortype::ComputeHash() const noexcept
{
  auto typeHash = typeid(fixedvectortype).hash_code();
  auto sizeHash = std::hash<size_t>()(size());
  return util::CombineHashes(typeHash, sizeHash, Type()->ComputeHash());
}

std::string
fixedvectortype::debug_string() const
{
  return util::strfmt("fixedvector[", type().debug_string(), ":", size(), "]");
}

/* scalablevectortype */

scalablevectortype::~scalablevectortype()
{}

bool
scalablevectortype::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  return vectortype::operator==(other);
}

std::size_t
scalablevectortype::ComputeHash() const noexcept
{
  auto typeHash = typeid(scalablevectortype).hash_code();
  auto sizeHash = std::hash<size_t>()(size());
  return util::CombineHashes(typeHash, sizeHash, Type()->ComputeHash());
}

std::string
scalablevectortype::debug_string() const
{
  return util::strfmt("scalablevector[", type().debug_string(), ":", size(), "]");
}

/* I/O state type */

iostatetype::~iostatetype()
{}

bool
iostatetype::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  return jlm::rvsdg::is<iostatetype>(other);
}

std::size_t
iostatetype::ComputeHash() const noexcept
{
  return typeid(iostatetype).hash_code();
}

std::string
iostatetype::debug_string() const
{
  return "iostate";
}

std::shared_ptr<const iostatetype>
iostatetype::Create()
{
  static const iostatetype instance;
  return std::shared_ptr<const iostatetype>(std::shared_ptr<void>(), &instance);
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
