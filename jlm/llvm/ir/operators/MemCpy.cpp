/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/MemCpy.hpp>

namespace jlm::llvm
{

MemCpyNonVolatileOperation::~MemCpyNonVolatileOperation() = default;

bool
MemCpyNonVolatileOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemCpyNonVolatileOperation *>(&other);
  return operation && operation->LengthType() == LengthType()
      && operation->NumMemoryStates() == NumMemoryStates();
}

std::string
MemCpyNonVolatileOperation::debug_string() const
{
  return "MemCpy";
}

std::unique_ptr<rvsdg::Operation>
MemCpyNonVolatileOperation::copy() const
{
  return std::make_unique<MemCpyNonVolatileOperation>(*this);
}

size_t
MemCpyNonVolatileOperation::NumMemoryStates() const noexcept
{
  return nresults();
}

MemCpyVolatileOperation::~MemCpyVolatileOperation() noexcept = default;

bool
MemCpyVolatileOperation::operator==(const Operation & other) const noexcept
{
  // Avoid common node elimination for memcpy operator
  return this == &other;
}

std::string
MemCpyVolatileOperation::debug_string() const
{
  return "MemCpyVolatile";
}

std::unique_ptr<rvsdg::Operation>
MemCpyVolatileOperation::copy() const
{
  return std::make_unique<MemCpyVolatileOperation>(*this);
}

size_t
MemCpyVolatileOperation::NumMemoryStates() const noexcept
{
  // Subtracting I/O state
  return nresults() - 1;
}

}
