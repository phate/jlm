/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
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
SExtOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
SExtOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  return nullptr;
}

std::optional<std::vector<rvsdg::Output *>>
SExtOperation::foldConstant(
    const SExtOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  auto & operand = *operands[0];

  const auto & tracedOperand = llvm::traceOutput(operand);
  auto [constantNode, constantOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(tracedOperand);
  if (!constantOperation)
    return std::nullopt;

  const auto & resultRepresentation =
      constantOperation->Representation().sext(operation.ndstbits() - operation.nsrcbits());

  auto result = IntegerConstantOperation::Create(*operand.region(), resultRepresentation).output(0);

  return std::vector<rvsdg::Output *>({ result });
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
ZExtOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
ZExtOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  return nullptr;
}

std::optional<std::vector<rvsdg::Output *>>
ZExtOperation::foldConstant(
    const ZExtOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  auto & operand = *operands[0];

  const auto & tracedOperand = llvm::traceOutput(operand);
  auto [constantNode, constantOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(tracedOperand);
  if (!constantOperation)
    return std::nullopt;

  const auto & resultRepresentation =
      constantOperation->Representation().zext(operation.ndstbits() - operation.nsrcbits());

  auto result = IntegerConstantOperation::Create(*operand.region(), resultRepresentation).output(0);

  return std::vector<rvsdg::Output *>({ result });
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

PtrToIntOperation::~PtrToIntOperation() noexcept = default;

bool
PtrToIntOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const PtrToIntOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
PtrToIntOperation::debug_string() const
{
  return "PtrToInt";
}

std::unique_ptr<rvsdg::Operation>
PtrToIntOperation::copy() const
{
  return std::make_unique<PtrToIntOperation>(*this);
}

rvsdg::unop_reduction_path_t
PtrToIntOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
PtrToIntOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

FPExtOperation::~FPExtOperation() noexcept = default;

bool
FPExtOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const FPExtOperation *>(&other);
  return op && op->srcsize() == srcsize() && op->dstsize() == dstsize();
}

std::string
FPExtOperation::debug_string() const
{
  return "FPExt";
}

std::unique_ptr<rvsdg::Operation>
FPExtOperation::copy() const
{
  return std::make_unique<FPExtOperation>(*this);
}

rvsdg::unop_reduction_path_t
FPExtOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
FPExtOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

FPTruncOperation::~FPTruncOperation() noexcept = default;

bool
FPTruncOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const FPTruncOperation *>(&other);
  return op && op->srcsize() == srcsize() && op->dstsize() == dstsize();
}

std::string
FPTruncOperation::debug_string() const
{
  return "FPTrunc";
}

std::unique_ptr<rvsdg::Operation>
FPTruncOperation::copy() const
{
  return std::make_unique<FPTruncOperation>(*this);
}

rvsdg::unop_reduction_path_t
FPTruncOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
FPTruncOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

UIToFPOperation::~UIToFPOperation() noexcept = default;

bool
UIToFPOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const UIToFPOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
UIToFPOperation::debug_string() const
{
  return "UIToFP";
}

std::unique_ptr<rvsdg::Operation>
UIToFPOperation::copy() const
{
  return std::make_unique<UIToFPOperation>(*this);
}

rvsdg::unop_reduction_path_t
UIToFPOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
UIToFPOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

SIToFPOperation::~SIToFPOperation() noexcept = default;

bool
SIToFPOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const SIToFPOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
SIToFPOperation::debug_string() const
{
  return "SIToFP";
}

std::unique_ptr<rvsdg::Operation>
SIToFPOperation::copy() const
{
  return std::make_unique<SIToFPOperation>(*this);
}

rvsdg::unop_reduction_path_t
SIToFPOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
SIToFPOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

IntegerToPointerOperation::~IntegerToPointerOperation() noexcept = default;

bool
IntegerToPointerOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const IntegerToPointerOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
IntegerToPointerOperation::debug_string() const
{
  return "IntToPtr";
}

std::unique_ptr<rvsdg::Operation>
IntegerToPointerOperation::copy() const
{
  return std::make_unique<IntegerToPointerOperation>(*this);
}

rvsdg::unop_reduction_path_t
IntegerToPointerOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
IntegerToPointerOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

FloatingPointToUnsignedIntegerOperation::~FloatingPointToUnsignedIntegerOperation() noexcept =
    default;

bool
FloatingPointToUnsignedIntegerOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const FloatingPointToUnsignedIntegerOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
FloatingPointToUnsignedIntegerOperation::debug_string() const
{
  return "FpToUInt";
}

std::unique_ptr<rvsdg::Operation>
FloatingPointToUnsignedIntegerOperation::copy() const
{
  return std::make_unique<FloatingPointToUnsignedIntegerOperation>(*this);
}

rvsdg::unop_reduction_path_t
FloatingPointToUnsignedIntegerOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
FloatingPointToUnsignedIntegerOperation::reduce_operand(
    rvsdg::unop_reduction_path_t,
    rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented");
}

FloatingPointToSignedIntegerOperation::~FloatingPointToSignedIntegerOperation() noexcept = default;

bool
FloatingPointToSignedIntegerOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const FloatingPointToSignedIntegerOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
FloatingPointToSignedIntegerOperation::debug_string() const
{
  return "FpToSInt";
}

std::unique_ptr<rvsdg::Operation>
FloatingPointToSignedIntegerOperation::copy() const
{
  return std::make_unique<FloatingPointToSignedIntegerOperation>(*this);
}

rvsdg::unop_reduction_path_t
FloatingPointToSignedIntegerOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
FloatingPointToSignedIntegerOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *)
    const
{
  JLM_UNREACHABLE("Not implemented!");
}

ControlToIntOperation::~ControlToIntOperation() noexcept = default;

bool
ControlToIntOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ControlToIntOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
ControlToIntOperation::debug_string() const
{
  return "ControlToInt";
}

std::unique_ptr<rvsdg::Operation>
ControlToIntOperation::copy() const
{
  return std::make_unique<ControlToIntOperation>(*this);
}
}
