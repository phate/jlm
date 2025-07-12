/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

namespace jlm::rvsdg
{

bitunary_op::~bitunary_op() noexcept
{}

unop_reduction_path_t
bitunary_op::can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept
{
  if (is<bitconstant_op>(producer(arg)))
    return unop_reduction_constant;

  return unop_reduction_none;
}

jlm::rvsdg::Output *
bitunary_op::reduce_operand(unop_reduction_path_t path, jlm::rvsdg::Output * arg) const
{
  if (path == unop_reduction_constant)
  {
    auto p = producer(arg);
    auto & c = static_cast<const bitconstant_op &>(p->GetOperation());
    return create_bitconstant(p->region(), reduce_constant(c.value()));
  }

  return nullptr;
}

BitBinaryOperation::~BitBinaryOperation() noexcept = default;

binop_reduction_path_t
BitBinaryOperation::can_reduce_operand_pair(
    const jlm::rvsdg::Output * arg1,
    const jlm::rvsdg::Output * arg2) const noexcept
{
  if (is<bitconstant_op>(producer(arg1)) && is<bitconstant_op>(producer(arg2)))
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
    auto & c1 = static_cast<const bitconstant_op &>(producer(arg1)->GetOperation());
    auto & c2 = static_cast<const bitconstant_op &>(producer(arg2)->GetOperation());
    return create_bitconstant(arg1->region(), reduce_constants(c1.value(), c2.value()));
  }

  return nullptr;
}

bitcompare_op::~bitcompare_op() noexcept
{}

binop_reduction_path_t
bitcompare_op::can_reduce_operand_pair(
    const jlm::rvsdg::Output * arg1,
    const jlm::rvsdg::Output * arg2) const noexcept
{
  auto p = producer(arg1);
  const bitconstant_op * c1_op = nullptr;
  if (p)
    c1_op = dynamic_cast<const bitconstant_op *>(&p->GetOperation());

  p = producer(arg2);
  const bitconstant_op * c2_op = nullptr;
  if (p)
    c2_op = dynamic_cast<const bitconstant_op *>(&p->GetOperation());

  bitvalue_repr arg1_repr = c1_op ? c1_op->value() : bitvalue_repr::repeat(type().nbits(), 'D');
  bitvalue_repr arg2_repr = c2_op ? c2_op->value() : bitvalue_repr::repeat(type().nbits(), 'D');

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
bitcompare_op::reduce_operand_pair(
    binop_reduction_path_t path,
    jlm::rvsdg::Output * arg1,
    jlm::rvsdg::Output *) const
{
  if (path == 1)
  {
    return create_bitconstant(arg1->region(), "0");
  }
  if (path == 2)
  {
    return create_bitconstant(arg1->region(), "1");
  }

  return nullptr;
}

}
