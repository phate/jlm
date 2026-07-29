/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/rvsdg/lambda.hpp>

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

static bool
isAllocationSide(rvsdg::Output & output)
{
  auto [allocaNode, allocaOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(output);
  if (allocaOperation)
  {
    return true;
  }

  if (rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(output))
  {
    return true;
  }

  if (dynamic_cast<const LlvmGraphImport *>(&output))
  {
    return true;
  }

  auto [fnToPtrNode, fnToPtrOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<FunctionToPointerOperation>(output);
  if (fnToPtrOperation != nullptr)
  {
    const auto & tracedOutput = rvsdg::traceOutput(*fnToPtrNode->input(0)->origin());
    if (rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(tracedOutput))
    {
      return true;
    }
  }

  return false;
}

std::optional<std::vector<rvsdg::Output *>>
IOBarrierOperation::normalizeDereferenceableAddressOperand(
    const IOBarrierOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 2);
  const auto operand = operands[0];

  if (!rvsdg::is<PointerType>(operand->Type()))
    return std::nullopt;

  auto & tracedOperand = rvsdg::traceOutput(*operand);
  if (!isAllocationSide(tracedOperand))
    return std::nullopt;

  return std::vector({ operand });
}

}
