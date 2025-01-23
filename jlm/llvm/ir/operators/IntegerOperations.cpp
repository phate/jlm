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
  return addOperation && addOperation->result(0) == result(0);
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
  return subOperation && subOperation->result(0) == result(0);
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
  return mulOperation && mulOperation->result(0) == result(0);
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
  return sdivOperation && sdivOperation->result(0) == result(0);
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
  return udivOperation && udivOperation->result(0) == result(0);
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
  return smodOperation && smodOperation->result(0) == result(0);
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
  return umodOperation && umodOperation->result(0) == result(0);
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
  return ashrOperation && ashrOperation->result(0) == result(0);
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
  return shlOperation && shlOperation->result(0) == result(0);
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
  return shrOperation && shrOperation->result(0) == result(0);
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
  return andOperation && andOperation->result(0) == result(0);
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
  return orOperation && orOperation->result(0) == result(0);
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
  return xorOperation && xorOperation->result(0) == result(0);
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

}
