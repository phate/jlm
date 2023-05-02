/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/types.hpp>

#include <unordered_map>

namespace jlm {

/**
 * FunctionType class
 */
FunctionType::~FunctionType() noexcept
= default;

FunctionType::FunctionType(
  const std::vector<const jive::type*> & argumentTypes,
  const std::vector<const jive::type*> & resultTypes)
  : jive::valuetype()
{
  for (auto & type : argumentTypes)
    ArgumentTypes_.emplace_back(type->copy());

  for (auto & type : resultTypes)
    ResultTypes_.emplace_back(type->copy());
}

FunctionType::FunctionType(
  std::vector<std::unique_ptr<jive::type>> argumentTypes,
  std::vector<std::unique_ptr<jive::type>> resultTypes)
  : jive::valuetype()
  , ResultTypes_(std::move(resultTypes))
  , ArgumentTypes_(std::move(argumentTypes))
{}

FunctionType::FunctionType(const FunctionType & rhs)
  : jive::valuetype(rhs)
{
  for (auto & type : rhs.ArgumentTypes_)
    ArgumentTypes_.push_back(type->copy());

  for (auto & type : rhs.ResultTypes_)
    ResultTypes_.push_back(type->copy());
}

FunctionType::FunctionType(FunctionType && other) noexcept
  : jive::valuetype(other)
  , ResultTypes_(std::move(other.ResultTypes_))
  , ArgumentTypes_(std::move(other.ArgumentTypes_))
{}

FunctionType::ArgumentConstRange
FunctionType::Arguments() const
{
  return {TypeConstIterator(ArgumentTypes_.begin()), TypeConstIterator(ArgumentTypes_.end())};
}

FunctionType::ResultConstRange
FunctionType::Results() const
{
  return {TypeConstIterator(ResultTypes_.begin()), TypeConstIterator(ResultTypes_.end())};
}

std::string
FunctionType::debug_string() const
{
  return "fct";
}

bool
FunctionType::operator==(const jive::type & _other) const noexcept
{
  auto other = dynamic_cast<const FunctionType*>(&_other);
  if (other == nullptr)
    return false;

  if (this->NumResults() != other->NumResults())
    return false;

  if (this->NumArguments() != other->NumArguments())
    return false;

  for (size_t i = 0; i < this->NumResults(); i++){
    if (this->ResultType(i) != other->ResultType(i))
      return false;
  }

  for (size_t i = 0; i < this->NumArguments(); i++){
    if (this->ArgumentType(i) != other->ArgumentType(i))
      return false;
  }

  return true;
}

std::unique_ptr<jive::type>
FunctionType::copy() const
{
  return std::unique_ptr<jive::type>(new FunctionType(*this));
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

PointerType::~PointerType() noexcept
= default;

std::string
PointerType::debug_string() const
{
  return "ptr";
}

bool
PointerType::operator==(const jive::type & other) const noexcept
{
  return jive::is<PointerType>(other);
}

std::unique_ptr<jive::type>
PointerType::copy() const
{
  return std::unique_ptr<jive::type>(new PointerType(*this));
}

/* array type */

arraytype::~arraytype()
{}

std::string
arraytype::debug_string() const
{
	return strfmt("[ ", nelements(), " x ", type_->debug_string(), " ]");
}

bool
arraytype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const jlm::arraytype*>(&other);
	return type && type->element_type() == element_type() && type->nelements() == nelements();
}

std::unique_ptr<jive::type>
arraytype::copy() const
{
	return std::unique_ptr<jive::type>(new arraytype(*this));
}

/* floating point type */

fptype::~fptype()
{}

std::string
fptype::debug_string() const
{
	static std::unordered_map<fpsize, std::string> map({
	  {fpsize::half, "half"}
	, {fpsize::flt, "float"}
	, {fpsize::dbl, "double"}
	, {fpsize::x86fp80, "x86fp80"}
	});

	JLM_ASSERT(map.find(size()) != map.end());
	return map[size()];
}

bool
fptype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const jlm::fptype*>(&other);
	return type && type->size() == size();
}

std::unique_ptr<jive::type>
fptype::copy() const
{
	return std::unique_ptr<jive::type>(new fptype(*this));
}

/* vararg type */

varargtype::~varargtype()
{}

bool
varargtype::operator==(const jive::type & other) const noexcept
{
	return dynamic_cast<const jlm::varargtype*>(&other) != nullptr;
}

std::string
varargtype::debug_string() const
{
	return "vararg";
}

std::unique_ptr<jive::type>
varargtype::copy() const
{
	return std::unique_ptr<jive::type>(new jlm::varargtype(*this));
}

StructType::~StructType()
= default;

bool
StructType::operator==(const jive::type & other) const noexcept
{
  auto type = dynamic_cast<const StructType*>(&other);
  return type
         && type->IsPacked_ == IsPacked_
         && type->Name_ == Name_
         && &type->Declaration_ == &Declaration_;
}

std::string
StructType::debug_string() const
{
  return "struct";
}

std::unique_ptr<jive::type>
StructType::copy() const
{
  return std::unique_ptr<jive::type>(new StructType(*this));
}

/* vectortype */

bool
vectortype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const vectortype*>(&other);
	return type
	    && type->size_ == size_
	    && *type->type_ == *type_;
}

/* fixedvectortype */

fixedvectortype::~fixedvectortype()
{}

bool
fixedvectortype::operator==(const jive::type & other) const noexcept
{
	return vectortype::operator==(other);
}

std::string
fixedvectortype::debug_string() const
{
	return strfmt("fixedvector[", type().debug_string(), ":", size(), "]");
}

std::unique_ptr<jive::type>
fixedvectortype::copy() const
{
	return std::unique_ptr<jive::type>(new fixedvectortype(*this));
}


/* scalablevectortype */

scalablevectortype::~scalablevectortype()
{}

bool
scalablevectortype::operator==(const jive::type & other) const noexcept
{
	return vectortype::operator==(other);
}

std::string
scalablevectortype::debug_string() const
{
	return strfmt("scalablevector[", type().debug_string(), ":", size(), "]");
}

std::unique_ptr<jive::type>
scalablevectortype::copy() const
{
	return std::unique_ptr<jive::type>(new scalablevectortype(*this));
}

/* loop state type */

loopstatetype::~loopstatetype()
{}

bool
loopstatetype::operator==(const jive::type & other) const noexcept
{
	return dynamic_cast<const loopstatetype*>(&other) != nullptr;
}

std::string
loopstatetype::debug_string() const
{
	return "loopstate";
}

std::unique_ptr<jive::type>
loopstatetype::copy() const
{
	return std::unique_ptr<jive::type>(new loopstatetype(*this));
}

/* I/O state type */

iostatetype::~iostatetype()
{}

bool
iostatetype::operator==(const jive::type & other) const noexcept
{
	return jive::is<iostatetype>(other);
}

std::string
iostatetype::debug_string() const
{
	return "iostate";
}

std::unique_ptr<jive::type>
iostatetype::copy() const
{
	return std::unique_ptr<jive::type>(new iostatetype(*this));
}

/**
 * MemoryStateType class
 */
MemoryStateType::~MemoryStateType() noexcept
= default;

std::string
MemoryStateType::debug_string() const
{
  return "mem";
}

bool
MemoryStateType::operator==(const jive::type &other) const noexcept
{
  return jive::is<MemoryStateType>(other);
}

std::unique_ptr<jive::type>
MemoryStateType::copy() const
{
  return std::unique_ptr<jive::type>(new MemoryStateType(*this));
}

}
