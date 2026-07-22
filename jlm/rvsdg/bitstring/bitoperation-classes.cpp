/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/Trace.hpp>

namespace jlm::rvsdg
{

BitUnaryOperation::~BitUnaryOperation() noexcept = default;

std::optional<std::vector<Output *>>
BitUnaryOperation::foldConstant(
    const BitUnaryOperation & operation,
    const std::vector<Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);

  const auto & tracedOperand = traceOutputIntraProcedurally(*operands[0]);
  auto [constantNode, constantOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand);
  if (constantOperation)
  {
    return std::vector({ &BitConstantOperation::create(
        *constantNode->region(),
        operation.reduce_constant(constantOperation->value())) });
  }

  return std::nullopt;
}

BitBinaryOperation::~BitBinaryOperation() noexcept = default;

binop_reduction_path_t
BitBinaryOperation::can_reduce_operand_pair(
    const jlm::rvsdg::Output * arg1,
    const jlm::rvsdg::Output * arg2) const noexcept
{
  auto & tracedOperand1 = traceOutputIntraProcedurally(*arg1);
  auto & tracedOperand2 = traceOutputIntraProcedurally(*arg2);

  if (rvsdg::IsOwnerNodeOperation<BitConstantOperation>(tracedOperand1)
      && rvsdg::IsOwnerNodeOperation<BitConstantOperation>(tracedOperand2))
    return binop_reduction_constants;

  return binop_reduction_none;
}

jlm::rvsdg::Output *
BitBinaryOperation::reduce_operand_pair(
    binop_reduction_path_t path,
    jlm::rvsdg::Output * arg1,
    jlm::rvsdg::Output * arg2) const
{
  if (path == binop_reduction_constants)
  {
    auto & tracedOperand1 = traceOutputIntraProcedurally(*arg1);
    auto [constantNode1, constantOperation1] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand1);

    auto & tracedOperand2 = traceOutputIntraProcedurally(*arg2);
    auto [constantNode2, constantOperation2] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand2);

    return &BitConstantOperation::create(
        *arg1->region(),
        reduce_constants(constantOperation1->value(), constantOperation2->value()));
  }

  return nullptr;
}

std::optional<std::vector<Output *>>
BitBinaryOperation::foldConstants(
    const BitBinaryOperation & operation,
    const std::vector<Output *> & operands)
{
  JLM_ASSERT(operands.size() == 2);
  auto & operand1 = *operands[0];
  auto & operand2 = *operands[1];

  const auto & tracedOperand1 = traceOutputIntraProcedurally(operand1);
  auto [constantNode1, constantOperation1] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand1);
  if (!constantOperation1)
  {
    return std::nullopt;
  }

  const auto & tracedOperand2 = traceOutputIntraProcedurally(operand2);
  auto [constantNode2, constantOperation2] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand2);
  if (!constantOperation2)
  {
    return std::nullopt;
  }

  auto & result = BitConstantOperation::create(
      *operand1.region(),
      operation.reduce_constants(constantOperation1->value(), constantOperation2->value()));
  return std::vector({ &result });
}

BitCompareOperation::~BitCompareOperation() noexcept = default;

binop_reduction_path_t
BitCompareOperation::can_reduce_operand_pair(const Output *, const Output *) const noexcept
{
  return binop_reduction_none;
}

Output *
BitCompareOperation::reduce_operand_pair(binop_reduction_path_t, Output *, Output *) const
{
  return nullptr;
}

std::optional<std::vector<Output *>>
BitCompareOperation::foldConstants(
    const BitCompareOperation & operation,
    const std::vector<Output *> & operands)
{
  JLM_ASSERT(operands.size() == 2);
  auto & operand1 = *operands[0];
  auto & operand2 = *operands[1];

  const auto & tracedOperand1 = traceOutputIntraProcedurally(operand1);
  auto [constantNode1, constantOperation1] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand1);

  const auto & tracedOperand2 = traceOutputIntraProcedurally(operand2);
  auto [constantNode2, constantOperation2] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand2);

  const BitValueRepresentation representation1 =
      constantOperation1 ? constantOperation1->value()
                         : BitValueRepresentation::repeat(operation.type().nbits(), 'D');
  const BitValueRepresentation representation2 =
      constantOperation2 ? constantOperation2->value()
                         : BitValueRepresentation::repeat(operation.type().nbits(), 'D');

  switch (operation.reduce_constants(representation1, representation2))
  {
  case compare_result::static_false:
    return std::vector(
        { &BitConstantOperation::create(*operand1.region(), BitValueRepresentation("0")) });
  case compare_result::static_true:
    return std::vector(
        { &BitConstantOperation::create(*operand1.region(), BitValueRepresentation("1")) });
  default:
    return std::nullopt;
  }
}

}
