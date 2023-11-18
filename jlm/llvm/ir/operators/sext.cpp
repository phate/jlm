/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>

namespace jlm::llvm
{

/* sext operation */

static const rvsdg::unop_reduction_path_t sext_reduction_bitunary = 128;
static const rvsdg::unop_reduction_path_t sext_reduction_bitbinary = 129;

static bool
is_bitunary_reducible(const rvsdg::output * operand)
{
  return rvsdg::is<rvsdg::bitunary_op>(rvsdg::node_output::node(operand));
}

static bool
is_bitbinary_reducible(const rvsdg::output * operand)
{
  return rvsdg::is<rvsdg::bitbinary_op>(rvsdg::node_output::node(operand));
}

static bool
is_inverse_reducible(const sext_op & op, const rvsdg::output * operand)
{
  auto node = rvsdg::node_output::node(operand);
  if (!node)
    return false;

  auto top = dynamic_cast<const trunc_op *>(&node->operation());
  return top && top->nsrcbits() == op.ndstbits();
}

static rvsdg::output *
perform_bitunary_reduction(const sext_op & op, rvsdg::output * operand)
{
  JLM_ASSERT(is_bitunary_reducible(operand));
  auto unary = rvsdg::node_output::node(operand);
  auto region = operand->region();
  auto uop = static_cast<const rvsdg::bitunary_op *>(&unary->operation());

  auto output = sext_op::create(op.ndstbits(), unary->input(0)->origin());
  return rvsdg::simple_node::create_normalized(region, *uop->create(op.ndstbits()), { output })[0];
}

static rvsdg::output *
perform_bitbinary_reduction(const sext_op & op, rvsdg::output * operand)
{
  JLM_ASSERT(is_bitbinary_reducible(operand));
  auto binary = rvsdg::node_output::node(operand);
  auto region = operand->region();
  auto bop = static_cast<const rvsdg::bitbinary_op *>(&binary->operation());

  JLM_ASSERT(binary->ninputs() == 2);
  auto op1 = sext_op::create(op.ndstbits(), binary->input(0)->origin());
  auto op2 = sext_op::create(op.ndstbits(), binary->input(1)->origin());

  return rvsdg::simple_node::create_normalized(
      region,
      *bop->create(op.ndstbits()),
      { op1, op2 })[0];
}

static rvsdg::output *
perform_inverse_reduction(const sext_op & op, rvsdg::output * operand)
{
  JLM_ASSERT(is_inverse_reducible(op, operand));
  return rvsdg::node_output::node(operand)->input(0)->origin();
}

sext_op::~sext_op()
{}

bool
sext_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const sext_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
sext_op::debug_string() const
{
  return util::strfmt("SEXT[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<rvsdg::operation>
sext_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new sext_op(*this));
}

rvsdg::unop_reduction_path_t
sext_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  if (rvsdg::is<rvsdg::bitconstant_op>(producer(operand)))
    return rvsdg::unop_reduction_constant;

  if (is_bitunary_reducible(operand))
    return sext_reduction_bitunary;

  if (is_bitbinary_reducible(operand))
    return sext_reduction_bitbinary;

  if (is_inverse_reducible(*this, operand))
    return rvsdg::unop_reduction_inverse;

  return rvsdg::unop_reduction_none;
}

rvsdg::output *
sext_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  if (path == rvsdg::unop_reduction_constant)
  {
    auto c = static_cast<const rvsdg::bitconstant_op *>(&producer(operand)->operation());
    return create_bitconstant(operand->region(), c->value().sext(ndstbits() - nsrcbits()));
  }

  if (path == sext_reduction_bitunary)
    return perform_bitunary_reduction(*this, operand);

  if (path == sext_reduction_bitbinary)
    return perform_bitbinary_reduction(*this, operand);

  if (path == rvsdg::unop_reduction_inverse)
    return perform_inverse_reduction(*this, operand);

  return nullptr;
}

}
