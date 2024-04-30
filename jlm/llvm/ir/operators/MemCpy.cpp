/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/MemCpy.hpp>

namespace jlm::llvm
{

Memcpy::~Memcpy() = default;

bool
Memcpy::operator==(const operation & other) const noexcept
{
  // Avoid common node elimination for memcpy operator
  return this == &other;
}

std::string
Memcpy::debug_string() const
{
  return "MemCpy";
}

std::unique_ptr<rvsdg::operation>
Memcpy::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new Memcpy(*this));
}

}
