/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/util/common.hpp>

namespace jlm::llvm
{

BitCastOperation::~BitCastOperation() noexcept = default;

bool
BitCastOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const BitCastOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
BitCastOperation::debug_string() const
{
  return util::strfmt(
      "BitCast[",
      argument(0)->debug_string(),
      " -> ",
      result(0)->debug_string(),
      "]");
}

std::unique_ptr<rvsdg::Operation>
BitCastOperation::copy() const
{
  return std::make_unique<BitCastOperation>(*this);
}

rvsdg::unop_reduction_path_t
BitCastOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
BitCastOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
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
  return util::strfmt("SExt[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<rvsdg::Operation>
SExtOperation::copy() const
{
  return std::make_unique<SExtOperation>(*this);
}

rvsdg::unop_reduction_path_t
SExtOperation::can_reduce_operand(const rvsdg::Output * operand) const noexcept
{
  auto & tracedOperand = llvm::traceOutput(*operand);
  if (rvsdg::IsOwnerNodeOperation<rvsdg::BitConstantOperation>(tracedOperand))
    return rvsdg::unop_reduction_constant;

  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
SExtOperation::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const
{
  if (path == rvsdg::unop_reduction_constant)
  {
    auto & tracedOutput = llvm::traceOutput(*operand);
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::BitConstantOperation>(tracedOutput);
    JLM_ASSERT(constantNode && constantOperation);
    return &rvsdg::BitConstantOperation::create(
        *operand->region(),
        constantOperation->value().sext(ndstbits() - nsrcbits()));
  }

  return nullptr;
}

ZExtOperation::~ZExtOperation() noexcept = default;

bool
ZExtOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const ZExtOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
ZExtOperation::debug_string() const
{
  return util::strfmt("ZExt[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<rvsdg::Operation>
ZExtOperation::copy() const
{
  return std::make_unique<ZExtOperation>(*this);
}

rvsdg::unop_reduction_path_t
ZExtOperation::can_reduce_operand(const rvsdg::Output * operand) const noexcept
{
  auto & tracedOperand = llvm::traceOutput(*operand);
  if (rvsdg::IsOwnerNodeOperation<rvsdg::BitConstantOperation>(tracedOperand))
    return rvsdg::unop_reduction_constant;

  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
ZExtOperation::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const
{
  if (path == rvsdg::unop_reduction_constant)
  {
    auto & tracedOperand = llvm::traceOutput(*operand);
    auto [_, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::BitConstantOperation>(tracedOperand);
    JLM_ASSERT(constantOperation);
    return &rvsdg::BitConstantOperation::create(
        *rvsdg::TryGetOwnerNode<rvsdg::Node>(*operand)->region(),
        constantOperation->value().zext(ndstbits() - nsrcbits()));
  }

  return nullptr;
}

TruncOperation::~TruncOperation() noexcept = default;

bool
TruncOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const TruncOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
TruncOperation::debug_string() const
{
  return util::strfmt("Trunc[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<rvsdg::Operation>
TruncOperation::copy() const
{
  return std::make_unique<TruncOperation>(*this);
}

rvsdg::unop_reduction_path_t
TruncOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
TruncOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

}
