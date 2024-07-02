/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-types.hpp"

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
valuetype::operator==(const rvsdg::type & other) const noexcept
{
  return dynamic_cast<const valuetype *>(&other) != nullptr;
}

std::size_t
valuetype::ComputeHash() const noexcept
{
  return std::hash<const valuetype *>()(GetInstance());
}

std::shared_ptr<const valuetype>
valuetype::Create()
{
  return std::shared_ptr<const valuetype>(std::shared_ptr<void>(), GetInstance());
}

const valuetype *
valuetype::GetInstance() noexcept
{
  static const valuetype instance;
  return &instance;
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
statetype::operator==(const rvsdg::type & other) const noexcept
{
  return dynamic_cast<const statetype *>(&other) != nullptr;
}

std::size_t
statetype::ComputeHash() const noexcept
{
  return std::hash<const statetype *>()(GetInstance());
}

std::shared_ptr<const statetype>
statetype::Create()
{
  return std::shared_ptr<const statetype>(std::shared_ptr<void>(), GetInstance());
}

const statetype *
statetype::GetInstance() noexcept
{
  static const statetype instance;
  return &instance;
}

}
