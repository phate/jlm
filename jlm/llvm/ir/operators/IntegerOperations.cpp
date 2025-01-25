/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>

namespace jlm::llvm
{

IntegerBinaryOperation::~IntegerBinaryOperation() = default;

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
IntegerAddOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerAddOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerSubOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerSubOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerMulOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerMulOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerSDivOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerSDivOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerUDivOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerUDivOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerSRemOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerSRemOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerURemOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerURemOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerAShrOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerAShrOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerShlOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerShlOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerLShrOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerLShrOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerAndOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerAndOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerOrOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerOrOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
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
IntegerXorOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerXorOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerXorOperation::flags() const noexcept
{
  return flags::associative | flags::commutative;
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
IntegerEqOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerEqOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerEqOperation::flags() const noexcept
{
  return flags::none;
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
IntegerNeOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerNeOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerNeOperation::flags() const noexcept
{
  return flags::none;
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
IntegerSgeOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerSgeOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSgeOperation::flags() const noexcept
{
  return flags::none;
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
IntegerSgtOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerSgtOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSgtOperation::flags() const noexcept
{
  return flags::none;
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
IntegerSleOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerSleOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSleOperation::flags() const noexcept
{
  return flags::none;
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
IntegerSltOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerSltOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerSltOperation::flags() const noexcept
{
  return flags::none;
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
IntegerUgeOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerUgeOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUgeOperation::flags() const noexcept
{
  return flags::none;
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
IntegerUgtOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerUgtOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUgtOperation::flags() const noexcept
{
  return flags::none;
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
IntegerUleOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerUleOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUleOperation::flags() const noexcept
{
  return flags::none;
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
IntegerUltOperation::can_reduce_operand_pair(const rvsdg::output *, const rvsdg::output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
IntegerUltOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::output *,
    rvsdg::output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
IntegerUltOperation::flags() const noexcept
{
  return flags::none;
}

}
