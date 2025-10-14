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
  return rvsdg::is<rvsdg::BitUnaryOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand));
}

static bool
is_bitbinary_reducible(const rvsdg::Output * operand)
{
  return rvsdg::is<rvsdg::BitBinaryOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand));
}

static bool
is_inverse_reducible(const SExtOperation & op, const rvsdg::Output * operand)
{
  const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand);
  if (!node)
    return false;

  const auto top = dynamic_cast<const TruncOperation *>(&node->GetOperation());
  return top && top->nsrcbits() == op.ndstbits();
}

static rvsdg::Output *
perform_bitunary_reduction(const SExtOperation & op, rvsdg::Output * operand)
{
  JLM_ASSERT(is_bitunary_reducible(operand));
  const auto unaryNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand);
  auto region = operand->region();
  auto uop = static_cast<const rvsdg::BitUnaryOperation *>(&unaryNode->GetOperation());

  auto output = SExtOperation::create(op.ndstbits(), unaryNode->input(0)->origin());
  std::unique_ptr<rvsdg::SimpleOperation> simpleOperation(
      util::assertedCast<rvsdg::SimpleOperation>(uop->create(op.ndstbits()).release()));
  return rvsdg::SimpleNode::Create(*region, std::move(simpleOperation), { output }).output(0);
}

static rvsdg::Output *
perform_bitbinary_reduction(const SExtOperation & op, rvsdg::Output * operand)
{
  JLM_ASSERT(is_bitbinary_reducible(operand));
  const auto binaryNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand);
  auto region = operand->region();
  auto bop = static_cast<const rvsdg::BitBinaryOperation *>(&binaryNode->GetOperation());

  JLM_ASSERT(binaryNode->ninputs() == 2);
  auto op1 = SExtOperation::create(op.ndstbits(), binaryNode->input(0)->origin());
  auto op2 = SExtOperation::create(op.ndstbits(), binaryNode->input(1)->origin());

  std::unique_ptr<rvsdg::SimpleOperation> simpleOperation(
      util::assertedCast<rvsdg::SimpleOperation>(bop->create(op.ndstbits()).release()));
  return rvsdg::SimpleNode::Create(*region, std::move(simpleOperation), { op1, op2 }).output(0);
}

static rvsdg::Output *
perform_inverse_reduction(const SExtOperation & op, rvsdg::Output * operand)
{
  JLM_ASSERT(is_inverse_reducible(op, operand));
  return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand)->input(0)->origin();
}

SExtOperation::~SExtOperation() noexcept = default;

bool
SExtOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const SExtOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
SExtOperation::debug_string() const
{
  return util::strfmt("SEXT[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<rvsdg::Operation>
SExtOperation::copy() const
{
  return std::make_unique<SExtOperation>(*this);
}

rvsdg::unop_reduction_path_t
SExtOperation::can_reduce_operand(const rvsdg::Output * operand) const noexcept
{
  auto & tracedOutput = rvsdg::TraceOutputIntraProcedurally(*operand);
  if (rvsdg::IsOwnerNodeOperation<rvsdg::bitconstant_op>(tracedOutput))
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
SExtOperation::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const
{
  if (path == rvsdg::unop_reduction_constant)
  {
    auto & tracedOutput = rvsdg::TraceOutputIntraProcedurally(*operand);
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::bitconstant_op>(tracedOutput);
    JLM_ASSERT(constantNode && constantOperation);
    return create_bitconstant(
        operand->region(),
        constantOperation->value().sext(ndstbits() - nsrcbits()));
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
