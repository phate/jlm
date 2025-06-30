/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-types.hpp"

#include <jlm/util/Hash.hpp>

namespace jlm::tests
{

/* valuetype */

valuetype::~valuetype()
{}

std::string
valuetype::debug_string() const
{
  return "valuetype";
}

bool
valuetype::operator==(const rvsdg::Type & other) const noexcept
{
  return dynamic_cast<const valuetype *>(&other) != nullptr;
}

std::size_t
valuetype::ComputeHash() const noexcept
{
  return typeid(valuetype).hash_code();
}

std::shared_ptr<const valuetype>
valuetype::Create()
{
  static const valuetype instance;
  return std::shared_ptr<const valuetype>(std::shared_ptr<void>(), &instance);
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
