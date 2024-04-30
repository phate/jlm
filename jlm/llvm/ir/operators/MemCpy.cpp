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

MemCpyVolatileOperation::~MemCpyVolatileOperation() noexcept = default;

bool
MemCpyVolatileOperation::operator==(const operation & other) const noexcept
{
  // Avoid common node elimination for memcpy operator
  return this == &other;
}

std::string
MemCpyVolatileOperation::debug_string() const
{
  return "MemCpyVolatile";
}

std::unique_ptr<rvsdg::operation>
MemCpyVolatileOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new MemCpyVolatileOperation(*this));
}

}
