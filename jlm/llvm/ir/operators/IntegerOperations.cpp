/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/Trace.hpp>

namespace jlm::llvm
{

template<typename TBinOp>
static IntegerValueRepresentation
foldBinaryOperation(const IntegerValueRepresentation & r1, const IntegerValueRepresentation & r2)
{
  static_assert(std::is_base_of_v<IntegerBinaryOperation, TBinOp>);

  if constexpr (std::is_same_v<TBinOp, IntegerEqOperation>)
    return IntegerValueRepresentation(r1.eq(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerNeOperation>)
    return IntegerValueRepresentation(r1.ne(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerSgeOperation>)
    return IntegerValueRepresentation(r1.sge(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerSgtOperation>)
    return IntegerValueRepresentation(r1.sgt(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerSleOperation>)
    return IntegerValueRepresentation(r1.sle(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerSltOperation>)
    return IntegerValueRepresentation(r1.slt(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerUgeOperation>)
    return IntegerValueRepresentation(r1.uge(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerUgtOperation>)
    return IntegerValueRepresentation(r1.ugt(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerUleOperation>)
    return IntegerValueRepresentation(r1.ule(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerUltOperation>)
    return IntegerValueRepresentation(r1.ult(r2));
  else if constexpr (std::is_same_v<TBinOp, IntegerXorOperation>)
    return IntegerValueRepresentation(r1.lxor(r2));
  else
    static_assert(sizeof(TBinOp) == 0, "Unsupported binary operation!");
}

/**
 * Performs constant folding on integer binary operations.
 *
 * @tparam TBinOp An integer binary operation
 * @param operands The operands of the integer binary operation.
 * @return A vector with a single element if constant folding succeeded, otherwise std::nullopt.
 */
template<typename TBinOp>
static std::optional<std::vector<rvsdg::Output *>>
foldBinaryOperationConstants(const std::vector<rvsdg::Output *> & operands)
{
  static_assert(std::is_base_of_v<IntegerBinaryOperation, TBinOp>);

  JLM_ASSERT(operands.size() == 2);
  auto & operand1 = *operands[0];
  auto & operand2 = *operands[1];

  const auto & tracedOperand1 = llvm::traceOutput(operand1);
  auto [c1Node, c1Operation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(tracedOperand1);
  if (!c1Operation)
    return std::nullopt;

  const auto & tracedOperand2 = llvm::traceOutput(operand2);
  auto [c2Node, c2Operation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(tracedOperand2);
  if (!c2Operation)
    return std::nullopt;

  auto & c1Representation = c1Operation->Representation();
  auto & c2Representation = c2Operation->Representation();
  const auto & resultRepresentation =
      foldBinaryOperation<TBinOp>(c1Representation, c2Representation);

  auto result =
      IntegerConstantOperation::Create(*operand1.region(), resultRepresentation).output(0);

  return std::vector<rvsdg::Output *>({ result });
}

IntegerConstantOperation::~IntegerConstantOperation() = default;

std::unique_ptr<rvsdg::Operation>
IntegerConstantOperation::copy() const
{
  return std::make_unique<IntegerConstantOperation>(*this);
}

std::string
IntegerConstantOperation::debug_string() const
{
  if (Representation().is_known() && Representation().nbits() <= 64)
    return util::strfmt("I", Representation().nbits(), "(", Representation().to_uint(), ")");

  return Representation().str();
}

bool
IntegerConstantOperation::operator==(const Operation & other) const noexcept
{
  const auto constant = dynamic_cast<const IntegerConstantOperation *>(&other);
  return constant && constant->Representation() == Representation();
}

IntegerBinaryOperation::~IntegerBinaryOperation() noexcept = default;

IntegerAddOperation::~IntegerAddOperation() noexcept = default;

bool
IntegerAddOperation::operator==(const Operation & other) const noexcept
{
  const auto addOperation = dynamic_cast<const IntegerAddOperation *>(&other);
  return addOperation && addOperation->Type() == Type();
}

std::string
IntegerAddOperation::debug_string() const
{
  return "IAdd";
}

std::unique_ptr<rvsdg::Operation>
IntegerAddOperation::copy() const
{
  return std::make_unique<IntegerAddOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerAddOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerAddOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerAddOperation::flags() const noexcept
{
  return flags::associative | flags::commutative;
}

IntegerSubOperation::~IntegerSubOperation() noexcept = default;

bool
IntegerSubOperation::operator==(const Operation & other) const noexcept
{
  const auto subOperation = dynamic_cast<const IntegerSubOperation *>(&other);
  return subOperation && subOperation->Type() == Type();
}

std::string
IntegerSubOperation::debug_string() const
{
  return "ISub";
}

std::unique_ptr<rvsdg::Operation>
IntegerSubOperation::copy() const
{
  return std::make_unique<IntegerSubOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerSubOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerSubOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSubOperation::flags() const noexcept
{
  return flags::none;
}

IntegerMulOperation::~IntegerMulOperation() noexcept = default;

bool
IntegerMulOperation::operator==(const Operation & other) const noexcept
{
  const auto mulOperation = dynamic_cast<const IntegerMulOperation *>(&other);
  return mulOperation && mulOperation->Type() == Type();
}

std::string
IntegerMulOperation::debug_string() const
{
  return "IMul";
}

std::unique_ptr<rvsdg::Operation>
IntegerMulOperation::copy() const
{
  return std::make_unique<IntegerMulOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerMulOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerMulOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerMulOperation::flags() const noexcept
{
  return flags::associative | flags::commutative;
}

IntegerSDivOperation::~IntegerSDivOperation() noexcept = default;

bool
IntegerSDivOperation::operator==(const Operation & other) const noexcept
{
  const auto sdivOperation = dynamic_cast<const IntegerSDivOperation *>(&other);
  return sdivOperation && sdivOperation->Type() == Type();
}

std::string
IntegerSDivOperation::debug_string() const
{
  return "ISDiv";
}

std::unique_ptr<rvsdg::Operation>
IntegerSDivOperation::copy() const
{
  return std::make_unique<IntegerSDivOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerSDivOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerSDivOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSDivOperation::flags() const noexcept
{
  return flags::none;
}

IntegerUDivOperation::~IntegerUDivOperation() noexcept = default;

bool
IntegerUDivOperation::operator==(const Operation & other) const noexcept
{
  const auto udivOperation = dynamic_cast<const IntegerUDivOperation *>(&other);
  return udivOperation && udivOperation->Type() == Type();
}

std::string
IntegerUDivOperation::debug_string() const
{
  return "IUDiv";
}

std::unique_ptr<rvsdg::Operation>
IntegerUDivOperation::copy() const
{
  return std::make_unique<IntegerUDivOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerUDivOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerUDivOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUDivOperation::flags() const noexcept
{
  return flags::none;
}

IntegerSRemOperation::~IntegerSRemOperation() noexcept = default;

bool
IntegerSRemOperation::operator==(const Operation & other) const noexcept
{
  const auto smodOperation = dynamic_cast<const IntegerSRemOperation *>(&other);
  return smodOperation && smodOperation->Type() == Type();
}

std::string
IntegerSRemOperation::debug_string() const
{
  return "ISMod";
}

std::unique_ptr<rvsdg::Operation>
IntegerSRemOperation::copy() const
{
  return std::make_unique<IntegerSRemOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerSRemOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerSRemOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSRemOperation::flags() const noexcept
{
  return flags::none;
}

IntegerURemOperation::~IntegerURemOperation() noexcept = default;

bool
IntegerURemOperation::operator==(const Operation & other) const noexcept
{
  const auto umodOperation = dynamic_cast<const IntegerURemOperation *>(&other);
  return umodOperation && umodOperation->Type() == Type();
}

std::string
IntegerURemOperation::debug_string() const
{
  return "IUMod";
}

std::unique_ptr<rvsdg::Operation>
IntegerURemOperation::copy() const
{
  return std::make_unique<IntegerURemOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerURemOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerURemOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerURemOperation::flags() const noexcept
{
  return flags::none;
}

IntegerAShrOperation::~IntegerAShrOperation() noexcept = default;

bool
IntegerAShrOperation::operator==(const Operation & other) const noexcept
{
  const auto ashrOperation = dynamic_cast<const IntegerAShrOperation *>(&other);
  return ashrOperation && ashrOperation->Type() == Type();
}

std::string
IntegerAShrOperation::debug_string() const
{
  return "IAShr";
}

std::unique_ptr<rvsdg::Operation>
IntegerAShrOperation::copy() const
{
  return std::make_unique<IntegerAShrOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerAShrOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerAShrOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerAShrOperation::flags() const noexcept
{
  return flags::none;
}

IntegerShlOperation::~IntegerShlOperation() noexcept = default;

bool
IntegerShlOperation::operator==(const Operation & other) const noexcept
{
  const auto shlOperation = dynamic_cast<const IntegerShlOperation *>(&other);
  return shlOperation && shlOperation->Type() == Type();
}

std::string
IntegerShlOperation::debug_string() const
{
  return "IShl";
}

std::unique_ptr<rvsdg::Operation>
IntegerShlOperation::copy() const
{
  return std::make_unique<IntegerShlOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerShlOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerShlOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerShlOperation::flags() const noexcept
{
  return flags::none;
}

IntegerLShrOperation::~IntegerLShrOperation() noexcept = default;

bool
IntegerLShrOperation::operator==(const Operation & other) const noexcept
{
  const auto shrOperation = dynamic_cast<const IntegerLShrOperation *>(&other);
  return shrOperation && shrOperation->Type() == Type();
}

std::string
IntegerLShrOperation::debug_string() const
{
  return "IShr";
}

std::unique_ptr<rvsdg::Operation>
IntegerLShrOperation::copy() const
{
  return std::make_unique<IntegerLShrOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerLShrOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerLShrOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerLShrOperation::flags() const noexcept
{
  return flags::none;
}

IntegerAndOperation::~IntegerAndOperation() noexcept = default;

bool
IntegerAndOperation::operator==(const Operation & other) const noexcept
{
  const auto andOperation = dynamic_cast<const IntegerAndOperation *>(&other);
  return andOperation && andOperation->Type() == Type();
}

std::string
IntegerAndOperation::debug_string() const
{
  return "IAnd";
}

std::unique_ptr<rvsdg::Operation>
IntegerAndOperation::copy() const
{
  return std::make_unique<IntegerAndOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerAndOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerAndOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerAndOperation::flags() const noexcept
{
  return flags::associative | flags::commutative;
}

IntegerOrOperation::~IntegerOrOperation() noexcept = default;

bool
IntegerOrOperation::operator==(const Operation & other) const noexcept
{
  const auto orOperation = dynamic_cast<const IntegerOrOperation *>(&other);
  return orOperation && orOperation->Type() == Type();
}

std::string
IntegerOrOperation::debug_string() const
{
  return "IOr";
}

std::unique_ptr<rvsdg::Operation>
IntegerOrOperation::copy() const
{
  return std::make_unique<IntegerOrOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerOrOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerOrOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerOrOperation::flags() const noexcept
{
  return flags::associative | flags::commutative;
}

IntegerXorOperation::~IntegerXorOperation() noexcept = default;

bool
IntegerXorOperation::operator==(const Operation & other) const noexcept
{
  const auto xorOperation = dynamic_cast<const IntegerXorOperation *>(&other);
  return xorOperation && xorOperation->Type() == Type();
}

std::string
IntegerXorOperation::debug_string() const
{
  return "IXor";
}

std::unique_ptr<rvsdg::Operation>
IntegerXorOperation::copy() const
{
  return std::make_unique<IntegerXorOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerXorOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerXorOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerXorOperation::flags() const noexcept
{
  return flags::associative | flags::commutative;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerXorOperation::foldConstants(
    const IntegerXorOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerXorOperation>(operands);
}

IntegerEqOperation::~IntegerEqOperation() noexcept = default;

bool
IntegerEqOperation::operator==(const Operation & other) const noexcept
{
  const auto eqOperation = dynamic_cast<const IntegerEqOperation *>(&other);
  return eqOperation && eqOperation->result(0) == result(0);
}

std::string
IntegerEqOperation::debug_string() const
{
  return "IEq";
}

std::unique_ptr<rvsdg::Operation>
IntegerEqOperation::copy() const
{
  return std::make_unique<IntegerEqOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerEqOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerEqOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerEqOperation::flags() const noexcept
{
  return flags::commutative;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerEqOperation::foldConstants(
    const IntegerEqOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerEqOperation>(operands);
}

IntegerNeOperation::~IntegerNeOperation() noexcept = default;

bool
IntegerNeOperation::operator==(const Operation & other) const noexcept
{
  const auto neOperation = dynamic_cast<const IntegerNeOperation *>(&other);
  return neOperation && neOperation->result(0) == result(0);
}

std::string
IntegerNeOperation::debug_string() const
{
  return "INe";
}

std::unique_ptr<rvsdg::Operation>
IntegerNeOperation::copy() const
{
  return std::make_unique<IntegerNeOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerNeOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerNeOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerNeOperation::flags() const noexcept
{
  return flags::commutative;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerNeOperation::foldConstants(
    const IntegerNeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerNeOperation>(operands);
}

IntegerSgeOperation::~IntegerSgeOperation() noexcept = default;

bool
IntegerSgeOperation::operator==(const Operation & other) const noexcept
{
  const auto sgeOperation = dynamic_cast<const IntegerSgeOperation *>(&other);
  return sgeOperation && sgeOperation->result(0) == result(0);
}

std::string
IntegerSgeOperation::debug_string() const
{
  return "ISge";
}

std::unique_ptr<rvsdg::Operation>
IntegerSgeOperation::copy() const
{
  return std::make_unique<IntegerSgeOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerSgeOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerSgeOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSgeOperation::flags() const noexcept
{
  return flags::none;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerSgeOperation::foldConstants(
    const IntegerSgeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerSgeOperation>(operands);
}

IntegerSgtOperation::~IntegerSgtOperation() noexcept = default;

bool
IntegerSgtOperation::operator==(const Operation & other) const noexcept
{
  const auto sgtOperation = dynamic_cast<const IntegerSgtOperation *>(&other);
  return sgtOperation && sgtOperation->result(0) == result(0);
}

std::string
IntegerSgtOperation::debug_string() const
{
  return "ISgt";
}

std::unique_ptr<rvsdg::Operation>
IntegerSgtOperation::copy() const
{
  return std::make_unique<IntegerSgtOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerSgtOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerSgtOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSgtOperation::flags() const noexcept
{
  return flags::none;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerSgtOperation::foldConstants(
    const IntegerSgtOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerSgtOperation>(operands);
}

IntegerSleOperation::~IntegerSleOperation() noexcept = default;

bool
IntegerSleOperation::operator==(const Operation & other) const noexcept
{
  const auto sleOperation = dynamic_cast<const IntegerSleOperation *>(&other);
  return sleOperation && sleOperation->result(0) == result(0);
}

std::string
IntegerSleOperation::debug_string() const
{
  return "ISle";
}

std::unique_ptr<rvsdg::Operation>
IntegerSleOperation::copy() const
{
  return std::make_unique<IntegerSleOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerSleOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerSleOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSleOperation::flags() const noexcept
{
  return flags::none;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerSleOperation::foldConstants(
    const IntegerSleOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerSleOperation>(operands);
}

IntegerSltOperation::~IntegerSltOperation() noexcept = default;

bool
IntegerSltOperation::operator==(const Operation & other) const noexcept
{
  const auto sltOperation = dynamic_cast<const IntegerSltOperation *>(&other);
  return sltOperation && sltOperation->result(0) == result(0);
}

std::string
IntegerSltOperation::debug_string() const
{
  return "ISlt";
}

std::unique_ptr<rvsdg::Operation>
IntegerSltOperation::copy() const
{
  return std::make_unique<IntegerSltOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerSltOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerSltOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSltOperation::flags() const noexcept
{
  return flags::none;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerSltOperation::foldConstants(
    const IntegerSltOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerSltOperation>(operands);
}

IntegerUgeOperation::~IntegerUgeOperation() noexcept = default;

bool
IntegerUgeOperation::operator==(const Operation & other) const noexcept
{
  const auto ugeOperation = dynamic_cast<const IntegerUgeOperation *>(&other);
  return ugeOperation && ugeOperation->result(0) == result(0);
}

std::string
IntegerUgeOperation::debug_string() const
{
  return "IUge";
}

std::unique_ptr<rvsdg::Operation>
IntegerUgeOperation::copy() const
{
  return std::make_unique<IntegerUgeOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerUgeOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerUgeOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUgeOperation::flags() const noexcept
{
  return flags::none;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerUgeOperation::foldConstants(
    const IntegerUgeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerUgeOperation>(operands);
}

IntegerUgtOperation::~IntegerUgtOperation() noexcept = default;

bool
IntegerUgtOperation::operator==(const Operation & other) const noexcept
{
  const auto ugtOperation = dynamic_cast<const IntegerUgtOperation *>(&other);
  return ugtOperation && ugtOperation->result(0) == result(0);
}

std::string
IntegerUgtOperation::debug_string() const
{
  return "IUgt";
}

std::unique_ptr<rvsdg::Operation>
IntegerUgtOperation::copy() const
{
  return std::make_unique<IntegerUgtOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerUgtOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerUgtOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUgtOperation::flags() const noexcept
{
  return flags::none;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerUgtOperation::foldConstants(
    const IntegerUgtOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerUgtOperation>(operands);
}

IntegerUleOperation::~IntegerUleOperation() noexcept = default;

bool
IntegerUleOperation::operator==(const Operation & other) const noexcept
{
  const auto uleOperation = dynamic_cast<const IntegerUleOperation *>(&other);
  return uleOperation && uleOperation->result(0) == result(0);
}

std::string
IntegerUleOperation::debug_string() const
{
  return "IUle";
}

std::unique_ptr<rvsdg::Operation>
IntegerUleOperation::copy() const
{
  return std::make_unique<IntegerUleOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerUleOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerUleOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUleOperation::flags() const noexcept
{
  return flags::none;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerUleOperation::foldConstants(
    const IntegerUleOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerUleOperation>(operands);
}

IntegerUltOperation::~IntegerUltOperation() noexcept = default;

bool
IntegerUltOperation::operator==(const Operation & other) const noexcept
{
  const auto ultOperation = dynamic_cast<const IntegerUltOperation *>(&other);
  return ultOperation && ultOperation->result(0) == result(0);
}

std::string
IntegerUltOperation::debug_string() const
{
  return "IUlt";
}

std::unique_ptr<rvsdg::Operation>
IntegerUltOperation::copy() const
{
  return std::make_unique<IntegerUltOperation>(*this);
}

rvsdg::binop_reduction_path_t
IntegerUltOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
IntegerUltOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUltOperation::flags() const noexcept
{
  return flags::none;
}

std::optional<std::vector<rvsdg::Output *>>
IntegerUltOperation::foldConstants(
    const IntegerUltOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  return foldBinaryOperationConstants<IntegerUltOperation>(operands);
}

}
