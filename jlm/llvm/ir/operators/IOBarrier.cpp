/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>

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

}
