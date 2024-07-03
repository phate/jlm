/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-types.hpp"

#include <jlm/util/HashSet.hpp>

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
  return util::ComputeConstantHash("jlm::tests::valuetype");
}

std::shared_ptr<const valuetype>
valuetype::Create()
{
  static const valuetype instance;
  return std::shared_ptr<const valuetype>(std::shared_ptr<void>(), &instance);
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
  return util::ComputeConstantHash("jlm::tests::statetype");
}

std::shared_ptr<const statetype>
statetype::Create()
{
  static const statetype instance;
  return std::shared_ptr<const statetype>(std::shared_ptr<void>(), &instance);
}

}
