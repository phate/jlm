/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/types.hpp>
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

FunctionType::FunctionType(const FunctionType & rhs)
    : jlm::rvsdg::valuetype(rhs)
{
  for (auto & type : rhs.ArgumentTypes_)
    ArgumentTypes_.push_back(type->copy());

  for (auto & type : rhs.ResultTypes_)
    ResultTypes_.push_back(type->copy());
}

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

std::shared_ptr<const jlm::rvsdg::type>
FunctionType::copy() const
{
  return std::make_shared<FunctionType>(*this);
}

FunctionType &
FunctionType::operator=(const FunctionType & rhs)
{
  ResultTypes_.clear();
  ArgumentTypes_.clear();

  for (auto & type : rhs.ArgumentTypes_)
    ArgumentTypes_.push_back(type->copy());

  for (auto & type : rhs.ResultTypes_)
    ResultTypes_.push_back(type->copy());

  return *this;
}

FunctionType &
FunctionType::operator=(FunctionType && rhs) noexcept
{
  ResultTypes_ = std::move(rhs.ResultTypes_);
  ArgumentTypes_ = std::move(rhs.ArgumentTypes_);
  return *this;
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

std::shared_ptr<const jlm::rvsdg::type>
PointerType::copy() const
{
  return std::make_shared<PointerType>(*this);
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

std::shared_ptr<const jlm::rvsdg::type>
arraytype::copy() const
{
  return std::make_shared<arraytype>(*this);
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

std::shared_ptr<const jlm::rvsdg::type>
fptype::copy() const
{
  return std::make_shared<fptype>(*this);
}

/* vararg type */

varargtype::~varargtype()
{}

bool
varargtype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  return dynamic_cast<const varargtype *>(&other) != nullptr;
}

std::string
varargtype::debug_string() const
{
  return "vararg";
}

std::shared_ptr<const jlm::rvsdg::type>
varargtype::copy() const
{
  return std::make_shared<varargtype>(*this);
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

std::string
StructType::debug_string() const
{
  return "struct";
}

std::shared_ptr<const jlm::rvsdg::type>
StructType::copy() const
{
  return std::make_shared<StructType>(*this);
}

/* vectortype */

bool
vectortype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  auto type = dynamic_cast<const vectortype *>(&other);
  return type && type->size_ == size_ && *type->type_ == *type_;
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

std::shared_ptr<const jlm::rvsdg::type>
fixedvectortype::copy() const
{
  return std::make_shared<fixedvectortype>(*this);
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

std::shared_ptr<const jlm::rvsdg::type>
scalablevectortype::copy() const
{
  return std::make_shared<scalablevectortype>(*this);
}

/* I/O state type */

iostatetype::~iostatetype()
{}

bool
iostatetype::operator==(const jlm::rvsdg::type & other) const noexcept
{
  return jlm::rvsdg::is<iostatetype>(other);
}

std::string
iostatetype::debug_string() const
{
  return "iostate";
}

std::shared_ptr<const jlm::rvsdg::type>
iostatetype::copy() const
{
  return std::make_shared<iostatetype>(*this);
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

std::shared_ptr<const jlm::rvsdg::type>
MemoryStateType::copy() const
{
  return std::make_shared<MemoryStateType>(*this);
}

std::shared_ptr<const MemoryStateType>
MemoryStateType::Create()
{
  static const MemoryStateType instance;
  return std::shared_ptr<const MemoryStateType>(std::shared_ptr<void>(), &instance);
}

}
