/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-types.hpp"

#include <jlm/util/Hash.hpp>

namespace jlm::tests
{

ValueType::~ValueType() noexcept = default;

std::string
ValueType::debug_string() const
{
  return "ValueType";
}

bool
ValueType::operator==(const Type & other) const noexcept
{
  return dynamic_cast<const ValueType *>(&other) != nullptr;
}

std::size_t
ValueType::ComputeHash() const noexcept
{
  return typeid(ValueType).hash_code();
}

std::shared_ptr<const ValueType>
ValueType::Create()
{
  static const ValueType instance;
  return std::shared_ptr<const ValueType>(std::shared_ptr<void>(), &instance);
}

/* statetype */

statetype::~statetype()
{}

std::string
statetype::debug_string() const
{
  return "statetype";
}

bool
statetype::operator==(const rvsdg::Type & other) const noexcept
{
  return dynamic_cast<const statetype *>(&other) != nullptr;
}

std::size_t
statetype::ComputeHash() const noexcept
{
  return typeid(statetype).hash_code();
}

std::shared_ptr<const statetype>
statetype::Create()
{
  static const statetype instance;
  return std::shared_ptr<const statetype>(std::shared_ptr<void>(), &instance);
}

}
