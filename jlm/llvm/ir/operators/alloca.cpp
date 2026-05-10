/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>

namespace jlm::llvm
{

AllocaOperation::~AllocaOperation() noexcept = default;

bool
AllocaOperation::operator==(const Operation & other) const noexcept
{
  // Avoid CNE for alloca operators
  return this == &other;
}

std::string
AllocaOperation::debug_string() const
{
  return "ALLOCA[" + allocatedType()->debug_string() + "]";
}

std::unique_ptr<rvsdg::Operation>
AllocaOperation::copy() const
{
  return std::make_unique<AllocaOperation>(*this);
}

}
