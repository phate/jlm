/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>

namespace jlm::llvm
{

static const rvsdg::unop_reduction_path_t sext_reduction_bitunary = 128;
static const rvsdg::unop_reduction_path_t sext_reduction_bitbinary = 129;

static bool
is_bitunary_reducible(const rvsdg::Output * operand)
{
  return rvsdg::is<rvsdg::bitunary_op>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand));
}

static bool
is_bitbinary_reducible(const rvsdg::Output * operand)
{
  return rvsdg::is<rvsdg::bitbinary_op>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand));
}

static bool
is_inverse_reducible(const sext_op & op, const rvsdg::Output * operand)
{
  const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand);
  if (!node)
    return false;

  const auto top = dynamic_cast<const TruncOperation *>(&node->GetOperation());
  return top && top->nsrcbits() == op.ndstbits();
}

static rvsdg::Output *
perform_bitunary_reduction(const sext_op & op, rvsdg::Output * operand)
{
  JLM_ASSERT(is_bitunary_reducible(operand));
  const auto unaryNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand);
  auto region = operand->region();
  auto uop = static_cast<const rvsdg::bitunary_op *>(&unaryNode->GetOperation());

  auto output = sext_op::create(op.ndstbits(), unaryNode->input(0)->origin());
  std::unique_ptr<rvsdg::SimpleOperation> simpleOperation(
      util::AssertedCast<rvsdg::SimpleOperation>(uop->create(op.ndstbits()).release()));
  return rvsdg::SimpleNode::Create(*region, std::move(simpleOperation), { output }).output(0);
}

static rvsdg::Output *
perform_bitbinary_reduction(const sext_op & op, rvsdg::Output * operand)
{
  JLM_ASSERT(is_bitbinary_reducible(operand));
  const auto binaryNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand);
  auto region = operand->region();
  auto bop = static_cast<const rvsdg::bitbinary_op *>(&binaryNode->GetOperation());

  JLM_ASSERT(binaryNode->ninputs() == 2);
  auto op1 = sext_op::create(op.ndstbits(), binaryNode->input(0)->origin());
  auto op2 = sext_op::create(op.ndstbits(), binaryNode->input(1)->origin());

  std::unique_ptr<rvsdg::SimpleOperation> simpleOperation(
      util::AssertedCast<rvsdg::SimpleOperation>(bop->create(op.ndstbits()).release()));
  return rvsdg::SimpleNode::Create(*region, std::move(simpleOperation), { op1, op2 }).output(0);
}

static rvsdg::Output *
perform_inverse_reduction(const sext_op & op, rvsdg::Output * operand)
{
  JLM_ASSERT(is_inverse_reducible(op, operand));
  return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand)->input(0)->origin();
}

sext_op::~sext_op()
{}

bool
sext_op::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const sext_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
sext_op::debug_string() const
{
  return util::strfmt("SEXT[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<rvsdg::Operation>
sext_op::copy() const
{
  return std::make_unique<sext_op>(*this);
}

rvsdg::unop_reduction_path_t
sext_op::can_reduce_operand(const rvsdg::Output * operand) const noexcept
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

rvsdg::Output *
sext_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const
{
  if (path == rvsdg::unop_reduction_constant)
  {
    auto c = static_cast<const rvsdg::bitconstant_op *>(&producer(operand)->GetOperation());
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
