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

StateType::~StateType() noexcept = default;

std::string
StateType::debug_string() const
{
  return "StateType";
}

bool
StateType::operator==(const Type & other) const noexcept
{
  return dynamic_cast<const StateType *>(&other) != nullptr;
}

std::size_t
StateType::ComputeHash() const noexcept
{
  return typeid(StateType).hash_code();
}

std::shared_ptr<const StateType>
StateType::Create()
{
  static const StateType instance;
  return std::shared_ptr<const StateType>(std::shared_ptr<void>(), &instance);
}

}
