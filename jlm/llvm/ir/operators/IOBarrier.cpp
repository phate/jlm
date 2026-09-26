/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/Trace.hpp>
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

std::optional<std::vector<rvsdg::Output *>>
MemoryHoistBarrierOperation::normalizeNestedMemoryHoistBarriers(
    const MemoryHoistBarrierOperation & lowerMhbOp,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 2);
  auto & lowerMhbAddressOperand = *operands[0];
  auto & lowerMhbIOStateOperand = *operands[1];

  auto & tracedLowerMhbAddressOperand = llvm::traceOutput(lowerMhbAddressOperand, true);
  auto [upperMhbNode, upperMhbOp] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryHoistBarrierOperation>(
          tracedLowerMhbAddressOperand);
  if (!upperMhbOp)
  {
    return std::nullopt;
  }
  auto & upperMhbAddressOperand = *getAddressInput(*upperMhbNode).origin();
  auto & upperMhbIOStateOperand = *getIOStateInput(*upperMhbNode).origin();

  auto & tracedUpperMhbIOStateOperand = llvm::traceOutput(upperMhbIOStateOperand, true);
  auto & tracedLowerMhbIOStateOperand = llvm::traceOutput(lowerMhbIOStateOperand, true);
  if (&tracedLowerMhbIOStateOperand != &tracedUpperMhbIOStateOperand)
  {
    return std::nullopt;
  }

  if (lowerMhbAddressOperand.region() != upperMhbAddressOperand.region())
  {
    return std::nullopt;
  }

  auto & newMhbNode = createNode(
      upperMhbAddressOperand,
      upperMhbIOStateOperand,
      std::min(lowerMhbOp.getDereferenceableSize(), upperMhbOp->getDereferenceableSize()));
  return rvsdg::outputs(&newMhbNode);
}

}
