/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/MemCpy.hpp>

namespace jlm::llvm
{

MemCpyOperation::~MemCpyOperation() = default;

bool
MemCpyOperation::operator==(const operation & other) const noexcept
{
  // Avoid common node elimination for memcpy operator
  return this == &other;
}

std::string
MemCpyOperation::debug_string() const
{
  return "MemCpy";
}

std::unique_ptr<rvsdg::operation>
MemCpyOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new MemCpyOperation(*this));
}

}
