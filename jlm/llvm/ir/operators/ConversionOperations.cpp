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

std::optional<std::vector<rvsdg::Output *>>
TruncOperation::foldConstant(
    const TruncOperation & operation,
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
      constantOperation->Representation().trunc(operation.ndstbits());

  auto result = IntegerConstantOperation::Create(*operand.region(), resultRepresentation).output(0);

  return std::vector<rvsdg::Output *>({ result });
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

IntToPtrOperation::~IntToPtrOperation() noexcept = default;

bool
IntToPtrOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const IntToPtrOperation *>(&other);
  return op && op->argument(0) == argument(0);
}

std::string
IntToPtrOperation::debug_string() const
{
  return "IntToPtr";
}

std::unique_ptr<rvsdg::Operation>
IntToPtrOperation::copy() const
{
  return std::make_unique<IntToPtrOperation>(*this);
}

FPToUIOperation::~FPToUIOperation() noexcept = default;

bool
FPToUIOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const FPToUIOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
FPToUIOperation::debug_string() const
{
  return "FpToUInt";
}

std::unique_ptr<rvsdg::Operation>
FPToUIOperation::copy() const
{
  return std::make_unique<FPToUIOperation>(*this);
}

FPToSIOperation::~FPToSIOperation() noexcept = default;

bool
FPToSIOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const FPToSIOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
FPToSIOperation::debug_string() const
{
  return "FpToSInt";
}

std::unique_ptr<rvsdg::Operation>
FPToSIOperation::copy() const
{
  return std::make_unique<FPToSIOperation>(*this);
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

FunctionToPointerOperation::~FunctionToPointerOperation() noexcept
{}

FunctionToPointerOperation::FunctionToPointerOperation(
    std::shared_ptr<const rvsdg::FunctionType> fn)
    : UnaryOperation(fn, PointerType::Create()),
      FunctionType_(std::move(fn))
{}

bool
FunctionToPointerOperation::operator==(const Operation & other) const noexcept
{
  if (auto o = dynamic_cast<const FunctionToPointerOperation *>(&other))
  {
    return *FunctionType() == *o->FunctionType();
  }
  else
  {
    return false;
  }
}

[[nodiscard]] std::string
FunctionToPointerOperation::debug_string() const
{
  return "FunPtr(" + FunctionType()->debug_string() + ")";
}

[[nodiscard]] std::unique_ptr<rvsdg::Operation>
FunctionToPointerOperation::copy() const
{
  return Create(FunctionType());
}

std::unique_ptr<FunctionToPointerOperation>
FunctionToPointerOperation::Create(std::shared_ptr<const rvsdg::FunctionType> fn)
{
  return std::make_unique<FunctionToPointerOperation>(std::move(fn));
}

std::optional<std::vector<rvsdg::Output *>>
FunctionToPointerOperation::invertFunctionToPointer(
    const FunctionToPointerOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  const auto & operand = *operands[0];

  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(operand))
  {
    if (const auto ptrToFnOperation =
            dynamic_cast<const PointerToFunctionOperation *>(&node->GetOperation()))
    {
      if (*ptrToFnOperation->FunctionType() == *operation.FunctionType())
      {
        return std::vector({ node->input(0)->origin() });
      }
    }
  }

  return std::nullopt;
}

PointerToFunctionOperation::~PointerToFunctionOperation() noexcept
{}

PointerToFunctionOperation::PointerToFunctionOperation(
    std::shared_ptr<const rvsdg::FunctionType> fn)
    : UnaryOperation(PointerType::Create(), fn),
      FunctionType_(std::move(fn))
{}

bool
PointerToFunctionOperation::operator==(const Operation & other) const noexcept
{
  if (auto o = dynamic_cast<const PointerToFunctionOperation *>(&other))
  {
    return *FunctionType() == *o->FunctionType();
  }
  else
  {
    return false;
  }
}

[[nodiscard]] std::string
PointerToFunctionOperation::debug_string() const
{
  return "PtrFun(" + FunctionType()->debug_string() + ")";
}

[[nodiscard]] std::unique_ptr<rvsdg::Operation>
PointerToFunctionOperation::copy() const
{
  return Create(FunctionType());
}

std::optional<std::vector<rvsdg::Output *>>
PointerToFunctionOperation::invertPointerToFunction(
    const PointerToFunctionOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  const auto & operand = *operands[0];

  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(operand))
  {
    if (const auto fnToPtrOperation =
            dynamic_cast<const FunctionToPointerOperation *>(&node->GetOperation()))
    {
      if (*fnToPtrOperation->FunctionType() == *operation.FunctionType())
      {
        return std::vector({ node->input(0)->origin() });
      }
    }
  }

  return std::nullopt;
}

std::unique_ptr<PointerToFunctionOperation>
PointerToFunctionOperation::Create(std::shared_ptr<const rvsdg::FunctionType> fn)
{
  return std::make_unique<PointerToFunctionOperation>(std::move(fn));
}

}
