/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/ControlOperations.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/Trace.hpp>

namespace jlm::llvm
{

std::optional<std::vector<rvsdg::Output *>>
foldMatchOperationWithConstant(
    const rvsdg::MatchOperation & matchOperation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  auto & operand = *operands[0];

  const auto & tracedOperand = llvm::traceOutput(operand);
  auto [constantNode, constantOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(tracedOperand);
  if (!constantOperation)
    return std::nullopt;

  const auto controlAlternative =
      matchOperation.alternative(constantOperation->Representation().to_uint());

  auto & controlConstantResult = rvsdg::ControlConstantOperation::create(
      *operand.region(),
      matchOperation.nalternatives(),
      controlAlternative);

  return std::vector({ &controlConstantResult });
}

}
