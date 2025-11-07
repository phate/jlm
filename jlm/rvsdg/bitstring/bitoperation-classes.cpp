/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

namespace jlm::rvsdg
{

BitUnaryOperation::~BitUnaryOperation() noexcept = default;

unop_reduction_path_t
BitUnaryOperation::can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept
{
  auto & tracedOperand = traceOutputIntraProcedurally(*arg);
  if (rvsdg::IsOwnerNodeOperation<BitConstantOperation>(tracedOperand))
    return unop_reduction_constant;

  return unop_reduction_none;
}

jlm::rvsdg::Output *
BitUnaryOperation::reduce_operand(unop_reduction_path_t path, jlm::rvsdg::Output * arg) const
{
  if (path == unop_reduction_constant)
  {
    auto & tracedOperand = traceOutputIntraProcedurally(*arg);
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand);
    return BitConstantOperation::create(
        constantNode->region(),
        reduce_constant(constantOperation->value()));
  }

  return nullptr;
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

    return BitConstantOperation::create(
        arg1->region(),
        reduce_constants(constantOperation1->value(), constantOperation2->value()));
  }

  return nullptr;
}

BitCompareOperation::~BitCompareOperation() noexcept = default;

binop_reduction_path_t
BitCompareOperation::can_reduce_operand_pair(
    const jlm::rvsdg::Output * arg1,
    const jlm::rvsdg::Output * arg2) const noexcept
{
  auto & tracedOperand1 = traceOutputIntraProcedurally(*arg1);
  auto [constantNode1, constantOperation1] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand1);
  auto & tracedOperand2 = traceOutputIntraProcedurally(*arg2);
  auto [constantNode2, constantOperation2] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(tracedOperand2);

  BitValueRepresentation arg1_repr = constantOperation1
                                       ? constantOperation1->value()
                                       : BitValueRepresentation::repeat(type().nbits(), 'D');
  BitValueRepresentation arg2_repr = constantOperation2
                                       ? constantOperation2->value()
                                       : BitValueRepresentation::repeat(type().nbits(), 'D');

  switch (reduce_constants(arg1_repr, arg2_repr))
  {
  case compare_result::static_false:
    return 1;
  case compare_result::static_true:
    return 2;
  case compare_result::undecidable:
    return binop_reduction_none;
  }

  return binop_reduction_none;
}

jlm::rvsdg::Output *
BitCompareOperation::reduce_operand_pair(
    binop_reduction_path_t path,
    jlm::rvsdg::Output * arg1,
    jlm::rvsdg::Output *) const
{
  if (path == 1)
  {
    return BitConstantOperation::create(arg1->region(), BitValueRepresentation("0"));
  }
  if (path == 2)
  {
    return BitConstantOperation::create(arg1->region(), BitValueRepresentation("1"));
  }

  return nullptr;
}

}
