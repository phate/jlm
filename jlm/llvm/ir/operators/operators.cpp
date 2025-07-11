/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

#include <llvm/ADT/SmallVector.h>

namespace jlm::llvm
{

SsaPhiOperation::~SsaPhiOperation() noexcept = default;

bool
SsaPhiOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const SsaPhiOperation *>(&other);
  return op && op->IncomingNodes_ == IncomingNodes_ && op->result(0) == result(0);
}

std::string
SsaPhiOperation::debug_string() const
{
  std::string str("[");
  for (size_t n = 0; n < narguments(); n++)
  {
    str += util::strfmt(GetIncomingNode(n));
    if (n != narguments() - 1)
      str += ", ";
  }
  str += "]";

  return "PHI" + str;
}

std::unique_ptr<rvsdg::Operation>
SsaPhiOperation::copy() const
{
  return std::make_unique<SsaPhiOperation>(*this);
}

AssignmentOperation::~AssignmentOperation() noexcept = default;

bool
AssignmentOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const AssignmentOperation *>(&other);
  return op && op->argument(0) == argument(0);
}

std::string
AssignmentOperation::debug_string() const
{
  return "ASSIGN";
}

std::unique_ptr<rvsdg::Operation>
AssignmentOperation::copy() const
{
  return std::make_unique<AssignmentOperation>(*this);
}

SelectOperation::~SelectOperation() noexcept = default;

bool
SelectOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const SelectOperation *>(&other);
  return op && op->result(0) == result(0);
}

std::string
SelectOperation::debug_string() const
{
  return "Select";
}

std::unique_ptr<rvsdg::Operation>
SelectOperation::copy() const
{
  return std::make_unique<SelectOperation>(*this);
}

VectorSelectOperation::~VectorSelectOperation() noexcept = default;

bool
VectorSelectOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const VectorSelectOperation *>(&other);
  return op && op->type() == type();
}

std::string
VectorSelectOperation::debug_string() const
{
  return "VectorSelect";
}

std::unique_ptr<rvsdg::Operation>
VectorSelectOperation::copy() const
{
  return std::make_unique<VectorSelectOperation>(*this);
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

/* ctl2bits operator */

ctl2bits_op::~ctl2bits_op() noexcept
{}

bool
ctl2bits_op::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ctl2bits_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
ctl2bits_op::debug_string() const
{
  return "CTL2BITS";
}

std::unique_ptr<rvsdg::Operation>
ctl2bits_op::copy() const
{
  return std::make_unique<ctl2bits_op>(*this);
}

BranchOperation::~BranchOperation() noexcept = default;

bool
BranchOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const BranchOperation *>(&other);
  return op && op->argument(0) == argument(0);
}

std::string
BranchOperation::debug_string() const
{
  return "Branch";
}

std::unique_ptr<rvsdg::Operation>
BranchOperation::copy() const
{
  return std::make_unique<BranchOperation>(*this);
}

ConstantPointerNullOperation::~ConstantPointerNullOperation() noexcept = default;

bool
ConstantPointerNullOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantPointerNullOperation *>(&other);
  return op && op->GetPointerType() == GetPointerType();
}

std::string
ConstantPointerNullOperation::debug_string() const
{
  return "ConstantPointerNull";
}

std::unique_ptr<rvsdg::Operation>
ConstantPointerNullOperation::copy() const
{
  return std::make_unique<ConstantPointerNullOperation>(*this);
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

ConstantDataArray::~ConstantDataArray()
{}

bool
ConstantDataArray::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantDataArray *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantDataArray::debug_string() const
{
  return "ConstantDataArray";
}

std::unique_ptr<rvsdg::Operation>
ConstantDataArray::copy() const
{
  return std::make_unique<ConstantDataArray>(*this);
}

/* pointer compare operator */

ptrcmp_op::~ptrcmp_op()
{}

bool
ptrcmp_op::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ptrcmp_op *>(&other);
  return op && op->argument(0) == argument(0) && op->cmp_ == cmp_;
}

std::string
ptrcmp_op::debug_string() const
{
  static std::unordered_map<llvm::cmp, std::string> map({ { cmp::eq, "eq" },
                                                          { cmp::ne, "ne" },
                                                          { cmp::gt, "gt" },
                                                          { cmp::ge, "ge" },
                                                          { cmp::lt, "lt" },
                                                          { cmp::le, "le" } });

  JLM_ASSERT(map.find(cmp()) != map.end());
  return "PTRCMP " + map[cmp()];
}

std::unique_ptr<rvsdg::Operation>
ptrcmp_op::copy() const
{
  return std::make_unique<ptrcmp_op>(*this);
}

rvsdg::binop_reduction_path_t
ptrcmp_op::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *) const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
ptrcmp_op::reduce_operand_pair(rvsdg::binop_reduction_path_t, rvsdg::Output *, rvsdg::Output *)
    const
{
  JLM_UNREACHABLE("Not implemented!");
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
  if (rvsdg::is<rvsdg::bitconstant_op>(producer(operand)))
    return rvsdg::unop_reduction_constant;

  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
ZExtOperation::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const
{
  if (path == rvsdg::unop_reduction_constant)
  {
    auto c = static_cast<const rvsdg::bitconstant_op *>(&producer(operand)->GetOperation());
    return create_bitconstant(
        rvsdg::TryGetOwnerNode<rvsdg::Node>(*operand)->region(),
        c->value().zext(ndstbits() - nsrcbits()));
  }

  return nullptr;
}

/* floating point constant operator */

ConstantFP::~ConstantFP()
{}

bool
ConstantFP::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantFP *>(&other);
  return op && size() == op->size() && constant().bitwiseIsEqual(op->constant());
}

std::string
ConstantFP::debug_string() const
{
  ::llvm::SmallVector<char, 32> v;
  constant().toString(v, 32, 0);

  std::string s("FP(");
  for (const auto & c : v)
    s += c;
  s += ")";

  return s;
}

std::unique_ptr<rvsdg::Operation>
ConstantFP::copy() const
{
  return std::make_unique<ConstantFP>(*this);
}

/* floating point comparison operator */

fpcmp_op::~fpcmp_op()
{}

bool
fpcmp_op::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const fpcmp_op *>(&other);
  return op && op->argument(0) == argument(0) && op->cmp_ == cmp_;
}

std::string
fpcmp_op::debug_string() const
{
  static std::unordered_map<fpcmp, std::string> map({ { fpcmp::oeq, "oeq" },
                                                      { fpcmp::ogt, "ogt" },
                                                      { fpcmp::oge, "oge" },
                                                      { fpcmp::olt, "olt" },
                                                      { fpcmp::ole, "ole" },
                                                      { fpcmp::one, "one" },
                                                      { fpcmp::ord, "ord" },
                                                      { fpcmp::ueq, "ueq" },
                                                      { fpcmp::ugt, "ugt" },
                                                      { fpcmp::uge, "uge" },
                                                      { fpcmp::ult, "ult" },
                                                      { fpcmp::ule, "ule" },
                                                      { fpcmp::une, "une" },
                                                      { fpcmp::uno, "uno" } });

  JLM_ASSERT(map.find(cmp()) != map.end());
  return "FPCMP " + map[cmp()];
}

std::unique_ptr<rvsdg::Operation>
fpcmp_op::copy() const
{
  return std::make_unique<fpcmp_op>(*this);
}

rvsdg::binop_reduction_path_t
fpcmp_op::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *) const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
fpcmp_op::reduce_operand_pair(rvsdg::binop_reduction_path_t, rvsdg::Output *, rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

UndefValueOperation::~UndefValueOperation() noexcept = default;

bool
UndefValueOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const UndefValueOperation *>(&other);
  return op && op->GetType() == GetType();
}

std::string
UndefValueOperation::debug_string() const
{
  return "undef";
}

std::unique_ptr<rvsdg::Operation>
UndefValueOperation::copy() const
{
  return std::make_unique<UndefValueOperation>(*this);
}

PoisonValueOperation::~PoisonValueOperation() noexcept = default;

bool
PoisonValueOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const PoisonValueOperation *>(&other);
  return operation && operation->GetType() == GetType();
}

std::string
PoisonValueOperation::debug_string() const
{
  return "poison";
}

std::unique_ptr<rvsdg::Operation>
PoisonValueOperation::copy() const
{
  return std::make_unique<PoisonValueOperation>(*this);
}

FBinaryOperation::~FBinaryOperation() noexcept = default;

bool
FBinaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const FBinaryOperation *>(&other);
  return op && op->fpop() == fpop() && op->size() == size();
}

std::string
FBinaryOperation::debug_string() const
{
  static std::unordered_map<llvm::fpop, std::string> map(
      { { fpop::add, "Add" },
        { fpop::sub, "Sub" },
        { fpop::mul, "Mul" },
        { fpop::div, "Div" },
        { fpop::mod, "Mod" } });

  JLM_ASSERT(map.find(fpop()) != map.end());
  return "F" + map[fpop()];
}

std::unique_ptr<rvsdg::Operation>
FBinaryOperation::copy() const
{
  return std::make_unique<FBinaryOperation>(*this);
}

rvsdg::binop_reduction_path_t
FBinaryOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
FBinaryOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
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

FNegOperation::~FNegOperation() noexcept = default;

bool
FNegOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const FNegOperation *>(&other);
  return op && op->size() == size();
}

std::string
FNegOperation::debug_string() const
{
  return "FNeg";
}

std::unique_ptr<rvsdg::Operation>
FNegOperation::copy() const
{
  return std::make_unique<FNegOperation>(*this);
}

rvsdg::unop_reduction_path_t
FNegOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
FNegOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
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

/* valist operator */

valist_op::~valist_op()
{}

bool
valist_op::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const valist_op *>(&other);
  if (!op || op->narguments() != narguments())
    return false;

  for (size_t n = 0; n < narguments(); n++)
  {
    if (op->argument(n) != argument(n))
      return false;
  }

  return true;
}

std::string
valist_op::debug_string() const
{
  return "VALIST";
}

std::unique_ptr<rvsdg::Operation>
valist_op::copy() const
{
  return std::make_unique<valist_op>(*this);
}

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

/* ConstantStruct operator */

ConstantStruct::~ConstantStruct()
{}

bool
ConstantStruct::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantStruct *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantStruct::debug_string() const
{
  return "ConstantStruct";
}

std::unique_ptr<rvsdg::Operation>
ConstantStruct::copy() const
{
  return std::make_unique<ConstantStruct>(*this);
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

ConstantArrayOperation::~ConstantArrayOperation() noexcept = default;

bool
ConstantArrayOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const ConstantArrayOperation *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantArrayOperation::debug_string() const
{
  return "ConstantArray";
}

std::unique_ptr<rvsdg::Operation>
ConstantArrayOperation::copy() const
{
  return std::make_unique<ConstantArrayOperation>(*this);
}

ConstantAggregateZeroOperation::~ConstantAggregateZeroOperation() noexcept = default;

bool
ConstantAggregateZeroOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const ConstantAggregateZeroOperation *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantAggregateZeroOperation::debug_string() const
{
  return "ConstantAggregateZero";
}

std::unique_ptr<rvsdg::Operation>
ConstantAggregateZeroOperation::copy() const
{
  return std::make_unique<ConstantAggregateZeroOperation>(*this);
}

ExtractElementOperation::~ExtractElementOperation() noexcept = default;

bool
ExtractElementOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ExtractElementOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->argument(1) == argument(1);
}

std::string
ExtractElementOperation::debug_string() const
{
  return "ExtractElement";
}

std::unique_ptr<rvsdg::Operation>
ExtractElementOperation::copy() const
{
  return std::make_unique<ExtractElementOperation>(*this);
}

ShuffleVectorOperation::~ShuffleVectorOperation() noexcept = default;

bool
ShuffleVectorOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ShuffleVectorOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->Mask() == Mask();
}

std::string
ShuffleVectorOperation::debug_string() const
{
  return "ShuffleVector";
}

std::unique_ptr<rvsdg::Operation>
ShuffleVectorOperation::copy() const
{
  return std::make_unique<ShuffleVectorOperation>(*this);
}

ConstantVectorOperation::~ConstantVectorOperation() noexcept = default;

bool
ConstantVectorOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantVectorOperation *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantVectorOperation::debug_string() const
{
  return "ConstantVector";
}

std::unique_ptr<rvsdg::Operation>
ConstantVectorOperation::copy() const
{
  return std::make_unique<ConstantVectorOperation>(*this);
}

InsertElementOperation::~InsertElementOperation() noexcept = default;

bool
InsertElementOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const InsertElementOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->argument(1) == argument(1)
      && op->argument(2) == argument(2);
}

std::string
InsertElementOperation::debug_string() const
{
  return "InsertElement";
}

std::unique_ptr<rvsdg::Operation>
InsertElementOperation::copy() const
{
  return std::make_unique<InsertElementOperation>(*this);
}

VectorUnaryOperation::~VectorUnaryOperation() noexcept = default;

bool
VectorUnaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const VectorUnaryOperation *>(&other);
  return op && op->operation() == operation();
}

std::string
VectorUnaryOperation::debug_string() const
{
  return util::strfmt("Vector", operation().debug_string());
}

std::unique_ptr<rvsdg::Operation>
VectorUnaryOperation::copy() const
{
  return std::make_unique<VectorUnaryOperation>(*this);
}

VectorBinaryOperation::~VectorBinaryOperation() noexcept = default;

bool
VectorBinaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const VectorBinaryOperation *>(&other);
  return op && op->operation() == operation();
}

std::string
VectorBinaryOperation::debug_string() const
{
  return util::strfmt("Vector", operation().debug_string());
}

std::unique_ptr<rvsdg::Operation>
VectorBinaryOperation::copy() const
{
  return std::make_unique<VectorBinaryOperation>(*this);
}

ConstantDataVectorOperation::~ConstantDataVectorOperation() noexcept = default;

bool
ConstantDataVectorOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantDataVectorOperation *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantDataVectorOperation::debug_string() const
{
  return "ConstantDataVector";
}

std::unique_ptr<rvsdg::Operation>
ConstantDataVectorOperation::copy() const
{
  return std::make_unique<ConstantDataVectorOperation>(*this);
}

ExtractValueOperation::~ExtractValueOperation() noexcept = default;

bool
ExtractValueOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ExtractValueOperation *>(&other);
  return op && op->indices_ == indices_ && op->type() == type();
}

std::string
ExtractValueOperation::debug_string() const
{
  return "ExtractValue";
}

std::unique_ptr<rvsdg::Operation>
ExtractValueOperation::copy() const
{
  return std::make_unique<ExtractValueOperation>(*this);
}

MallocOperation::~MallocOperation() noexcept = default;

bool
MallocOperation::operator==(const Operation & other) const noexcept
{
  // Avoid CNE for malloc operator
  return this == &other;
}

std::string
MallocOperation::debug_string() const
{
  return "Malloc";
}

std::unique_ptr<rvsdg::Operation>
MallocOperation::copy() const
{
  return std::make_unique<MallocOperation>(*this);
}

/* free operator */

FreeOperation::~FreeOperation() noexcept = default;

bool
FreeOperation::operator==(const Operation & other) const noexcept
{
  // Avoid CNE for free operator
  return this == &other;
}

std::string
FreeOperation::debug_string() const
{
  return "FREE";
}

std::unique_ptr<rvsdg::Operation>
FreeOperation::copy() const
{
  return std::make_unique<FreeOperation>(*this);
}

}
