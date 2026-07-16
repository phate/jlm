/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/util/BijectiveMap.hpp>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/InstrTypes.h>
#include <stdexcept>

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
  return dynamic_cast<const ConstantPointerNullOperation *>(&other);
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

ConstantDataArrayOperation::~ConstantDataArrayOperation() noexcept = default;

bool
ConstantDataArrayOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const ConstantDataArrayOperation *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantDataArrayOperation::debug_string() const
{
  return "ConstantDataArray";
}

std::unique_ptr<rvsdg::Operation>
ConstantDataArrayOperation::copy() const
{
  return std::make_unique<ConstantDataArrayOperation>(*this);
}

static const util::BijectiveMap<::llvm::CmpInst::Predicate, ICmpPredicate> &
getICmpPredicateMap()
{
  static util::BijectiveMap<::llvm::CmpInst::Predicate, ICmpPredicate> map = {
    { ::llvm::CmpInst::ICMP_EQ, ICmpPredicate::Eq },
    { ::llvm::CmpInst::ICMP_NE, ICmpPredicate::Ne },
    { ::llvm::CmpInst::ICMP_UGT, ICmpPredicate::Ugt },
    { ::llvm::CmpInst::ICMP_UGE, ICmpPredicate::Uge },
    { ::llvm::CmpInst::ICMP_ULT, ICmpPredicate::Ult },
    { ::llvm::CmpInst::ICMP_ULE, ICmpPredicate::Ule },
    { ::llvm::CmpInst::ICMP_SGT, ICmpPredicate::Sgt },
    { ::llvm::CmpInst::ICMP_SGE, ICmpPredicate::Sge },
    { ::llvm::CmpInst::ICMP_SLT, ICmpPredicate::Slt },
    { ::llvm::CmpInst::ICMP_SLE, ICmpPredicate::Sle },
  };
  return map;
}

ICmpPredicate
convertICmpPredicateToJlm(::llvm::CmpInst::Predicate predicate)
{
  const auto & map = getICmpPredicateMap();
  return map.LookupKey(predicate);
}

[[nodiscard]] ::llvm::CmpInst::Predicate
convertICmpPredicateToLlvm(ICmpPredicate predicate)
{
  const auto & map = getICmpPredicateMap();
  return map.LookupValue(predicate);
}

[[nodiscard]] std::string_view
iCmpPredicateToString(ICmpPredicate predicate)
{
  switch (predicate)
  {
  case ICmpPredicate::Eq:
    return "eq";
  case ICmpPredicate::Ne:
    return "ne";
  case ICmpPredicate::Ugt:
    return "ugt";
  case ICmpPredicate::Uge:
    return "uge";
  case ICmpPredicate::Ult:
    return "ult";
  case ICmpPredicate::Ule:
    return "ule";
  case ICmpPredicate::Sgt:
    return "sgt";
  case ICmpPredicate::Sge:
    return "sge";
  case ICmpPredicate::Slt:
    return "slt";
  case ICmpPredicate::Sle:
    return "sle";
  default:
    throw std::runtime_error("Unknown ICmpPredicate");
  }
}

PtrCmpOperation::~PtrCmpOperation() noexcept = default;

bool
PtrCmpOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const PtrCmpOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->predicate_ == predicate_;
}

std::string
PtrCmpOperation::debug_string() const
{
  return util::strfmt("PtrCmp[", iCmpPredicateToString(predicate_), "]");
}

std::unique_ptr<rvsdg::Operation>
PtrCmpOperation::copy() const
{
  return std::make_unique<PtrCmpOperation>(*this);
}

rvsdg::binop_reduction_path_t
PtrCmpOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
PtrCmpOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  JLM_UNREACHABLE("Not implemented!");
}

template<typename TOperation>
static bool
isOperand(rvsdg::Output & operand)
{
  auto [node, operation] = rvsdg::TryGetSimpleNodeAndOptionalOp<TOperation>(operand);
  return operation != nullptr;
}

std::optional<std::vector<rvsdg::Output *>>
PtrCmpOperation::normalizeNullPointerComparison(
    const PtrCmpOperation & ptrCmpOperation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (ptrCmpOperation.predicate() != ICmpPredicate::Eq
      && ptrCmpOperation.predicate() != ICmpPredicate::Ne)
    return std::nullopt;

  JLM_ASSERT(operands.size() == 2);
  auto & tracedOperand1 = rvsdg::traceOutputIntraProcedurally(*operands[0]);
  auto & tracedOperand2 = rvsdg::traceOutputIntraProcedurally(*operands[1]);

  const bool hasRequiredOperandProducers =
      (isOperand<ConstantPointerNullOperation>(tracedOperand1)
       && (isOperand<AllocaOperation>(tracedOperand2)
           || rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(tracedOperand2)))
      || (isOperand<ConstantPointerNullOperation>(tracedOperand2)
          && (isOperand<AllocaOperation>(tracedOperand1)
              || rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(tracedOperand1)));
  if (!hasRequiredOperandProducers)
  {
    return std::nullopt;
  }

  JLM_ASSERT(0 && "We found a valid transformation");
}

ConstantFP::~ConstantFP() noexcept = default;

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

FCmpOperation::~FCmpOperation() noexcept = default;

bool
FCmpOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const FCmpOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->cmp_ == cmp_;
}

std::string
FCmpOperation::debug_string() const
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
  return "FCmp " + map[cmp()];
}

std::unique_ptr<rvsdg::Operation>
FCmpOperation::copy() const
{
  return std::make_unique<FCmpOperation>(*this);
}

rvsdg::binop_reduction_path_t
FCmpOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *) const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
FCmpOperation::reduce_operand_pair(rvsdg::binop_reduction_path_t, rvsdg::Output *, rvsdg::Output *)
    const
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

FreezeOperation::~FreezeOperation() noexcept = default;

bool
FreezeOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const FreezeOperation *>(&other);
  return operation && operation->getType() == getType();
}

rvsdg::unop_reduction_path_t
FreezeOperation::can_reduce_operand([[maybe_unused]] const jlm::rvsdg::Output * arg) const noexcept
{
  return rvsdg::unop_reduction_none;
}

jlm::rvsdg::Output *
FreezeOperation::reduce_operand(
    [[maybe_unused]] rvsdg::unop_reduction_path_t path,
    [[maybe_unused]] jlm::rvsdg::Output * arg) const
{
  throw std::runtime_error("FreezeOperation does not support reductions");
}

std::string
FreezeOperation::debug_string() const
{
  return "freeze";
}

std::unique_ptr<rvsdg::Operation>
FreezeOperation::copy() const
{
  return std::make_unique<FreezeOperation>(*this);
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
  static std::unordered_map<llvm::fpop, std::string> map({ { fpop::add, "add" },
                                                           { fpop::sub, "sub" },
                                                           { fpop::mul, "mul" },
                                                           { fpop::div, "div" },
                                                           { fpop::mod, "mod" } });

  JLM_ASSERT(map.find(fpop()) != map.end());
  return "FPOP " + map[fpop()];
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

VariadicArgumentListOperation::~VariadicArgumentListOperation() noexcept = default;

bool
VariadicArgumentListOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const VariadicArgumentListOperation *>(&other);
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
VariadicArgumentListOperation::debug_string() const
{
  return "VariadicArguments";
}

std::unique_ptr<rvsdg::Operation>
VariadicArgumentListOperation::copy() const
{
  return std::make_unique<VariadicArgumentListOperation>(*this);
}

ConstantStructOperation::~ConstantStructOperation() noexcept = default;

bool
ConstantStructOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantStructOperation *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantStructOperation::debug_string() const
{
  return "ConstantStruct";
}

std::unique_ptr<rvsdg::Operation>
ConstantStructOperation::copy() const
{
  return std::make_unique<ConstantStructOperation>(*this);
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
