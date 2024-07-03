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

/**
 * FunctionType class
 */
FunctionType::~FunctionType() noexcept = default;

FunctionType::FunctionType(
    std::vector<std::shared_ptr<const jlm::rvsdg::type>> argumentTypes,
    std::vector<std::shared_ptr<const jlm::rvsdg::type>> resultTypes)
    : jlm::rvsdg::valuetype(),
      ResultTypes_(std::move(resultTypes)),
      ArgumentTypes_(std::move(argumentTypes))
{}

FunctionType::FunctionType(const FunctionType & rhs) = default;

FunctionType::FunctionType(FunctionType && other) noexcept
    : jlm::rvsdg::valuetype(other),
      ResultTypes_(std::move(other.ResultTypes_)),
      ArgumentTypes_(std::move(other.ArgumentTypes_))
{}

const std::vector<std::shared_ptr<const jlm::rvsdg::type>> &
FunctionType::Arguments() const noexcept
{
  return ArgumentTypes_;
}

const std::vector<std::shared_ptr<const jlm::rvsdg::type>> &
FunctionType::Results() const noexcept
{
  return ResultTypes_;
}

std::string
FunctionType::debug_string() const
{
  return "fct";
}

bool
FunctionType::operator==(const jlm::rvsdg::type & _other) const noexcept
{
  auto other = dynamic_cast<const FunctionType *>(&_other);
  if (other == nullptr)
    return false;

  if (this->NumResults() != other->NumResults())
    return false;

  if (this->NumArguments() != other->NumArguments())
    return false;

  for (size_t i = 0; i < this->NumResults(); i++)
  {
    if (this->ResultType(i) != other->ResultType(i))
      return false;
  }

  for (size_t i = 0; i < this->NumArguments(); i++)
  {
    if (this->ArgumentType(i) != other->ArgumentType(i))
      return false;
  }

  return true;
}

std::size_t
FunctionType::ComputeHash() const noexcept
{
  std::size_t seed = 0;
  for (auto argumentType : ArgumentTypes_)
  {
    util::CombineHashesWithSeed(seed, argumentType->ComputeHash());
  }
  for (auto resultType : ResultTypes_)
  {
    util::CombineHashesWithSeed(seed, resultType->ComputeHash());
  }

  return seed;
}

FunctionType &
FunctionType::operator=(const FunctionType & rhs) = default;

FunctionType &
FunctionType::operator=(FunctionType && rhs) noexcept
{
  ResultTypes_ = std::move(rhs.ResultTypes_);
  ArgumentTypes_ = std::move(rhs.ArgumentTypes_);
  return *this;
}

std::shared_ptr<const FunctionType>
FunctionType::Create(
    std::vector<std::shared_ptr<const jlm::rvsdg::type>> argumentTypes,
    std::vector<std::shared_ptr<const jlm::rvsdg::type>> resultTypes)
{
  return std::make_shared<FunctionType>(std::move(argumentTypes), std::move(resultTypes));
}

PointerType::~PointerType() noexcept = default;

std::string
PointerType::debug_string() const
{
  return "ptr";
}

bool
PointerType::operator==(const jlm::rvsdg::type & other) const noexcept
{
  return jlm::rvsdg::is<PointerType>(other);
}

std::size_t
PointerType::ComputeHash() const noexcept
{
  return util::ComputeConstantHash("jlm::llvm::PointerType");
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
arraytype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  auto type = dynamic_cast<const arraytype *>(&other);
  return type && type->element_type() == element_type() && type->nelements() == nelements();
}

std::size_t
arraytype::ComputeHash() const noexcept
{
  auto numElementsHash = std::hash<std::size_t>()(nelements_);
  return util::CombineHashes(type_->ComputeHash(), numElementsHash);
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
                                                       { fpsize::x86fp80, "x86fp80" } });

  JLM_ASSERT(map.find(size()) != map.end());
  return map[size()];
}

bool
fptype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  auto type = dynamic_cast<const fptype *>(&other);
  return type && type->size() == size();
}

std::size_t
fptype::ComputeHash() const noexcept
{
  return std::hash<fpsize>()(size_);
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
varargtype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  return dynamic_cast<const varargtype *>(&other) != nullptr;
}

std::size_t
varargtype::ComputeHash() const noexcept
{
  return util::ComputeConstantHash("jlm::llvm::varargtype");
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
StructType::operator==(const jlm::rvsdg::type & other) const noexcept
{
  auto type = dynamic_cast<const StructType *>(&other);
  return type && type->IsPacked_ == IsPacked_ && type->Name_ == Name_
      && &type->Declaration_ == &Declaration_;
}

std::size_t
StructType::ComputeHash() const noexcept
{
  auto isPackedHash = std::hash<bool>()(IsPacked_);
  auto nameHash = std::hash<std::string>()(Name_);
  auto declarationHash = std::hash<const StructType::Declaration *>()(&Declaration_);
  return util::CombineHashes(isPackedHash, nameHash, declarationHash);
}

std::string
StructType::debug_string() const
{
  return "struct";
}

/* vectortype */

bool
vectortype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  auto type = dynamic_cast<const vectortype *>(&other);
  return type && type->size_ == size_ && *type->type_ == *type_;
}

std::size_t
vectortype::ComputeHash() const noexcept
{
  auto sizeHash = std::hash<size_t>()(size_);
  return util::CombineHashes(sizeHash, type_->ComputeHash());
}

/* fixedvectortype */

fixedvectortype::~fixedvectortype()
{}

bool
fixedvectortype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  return vectortype::operator==(other);
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
scalablevectortype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  return vectortype::operator==(other);
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
iostatetype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  return jlm::rvsdg::is<iostatetype>(other);
}

std::size_t
iostatetype::ComputeHash() const noexcept
{
  return util::ComputeConstantHash("jlm::llvm::iostatetype");
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
MemoryStateType::operator==(const jlm::rvsdg::type & other) const noexcept
{
  return jlm::rvsdg::is<MemoryStateType>(other);
}

std::size_t
MemoryStateType::ComputeHash() const noexcept
{
  return util::ComputeConstantHash("jlm::llvm::MemoryStateType");
}

std::shared_ptr<const MemoryStateType>
MemoryStateType::Create()
{
  static const MemoryStateType instance;
  return std::shared_ptr<const MemoryStateType>(std::shared_ptr<void>(), &instance);
}

}
