/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::llvm
{

IOBarrierOperation::~IOBarrierOperation() noexcept = default;

bool
IOBarrierOperation::operator==(const Operation & other) const noexcept
{
  const auto ioBarrier = dynamic_cast<const IOBarrierOperation *>(&other);
  return ioBarrier && ioBarrier->Type() == Type();
}

std::string
IOBarrierOperation::debug_string() const
{
  return "IOBarrier";
}

std::unique_ptr<rvsdg::Operation>
IOBarrierOperation::copy() const
{
  return std::make_unique<IOBarrierOperation>(*this);
}

MemoryHoistBarrierOperation::~MemoryHoistBarrierOperation() noexcept = default;

bool
MemoryHoistBarrierOperation::operator==(const Operation & other) const noexcept
{
  const auto hoistBarrier = dynamic_cast<const MemoryHoistBarrierOperation *>(&other);
  return hoistBarrier && hoistBarrier->getDereferenceableSize() == getDereferenceableSize();
}

std::string
MemoryHoistBarrierOperation::debug_string() const
{
  return util::strfmt("MemoryHoistBarrier[", getDereferenceableSize(), "]");
}

std::unique_ptr<rvsdg::Operation>
MemoryHoistBarrierOperation::copy() const
{
  return std::make_unique<MemoryHoistBarrierOperation>(*this);
}

}
