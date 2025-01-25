/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/nullary.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::llvm
{

// FIXME: Implement our own value representation instead of re-using the bitstring value
// representation
using IntegerValueRepresentation = rvsdg::bitvalue_repr;

// FIXME: add documentation
class IntegerConstantOperation final : public rvsdg::NullaryOperation
{
public:
  ~IntegerConstantOperation() override;

  explicit IntegerConstantOperation(IntegerValueRepresentation representation)
      : NullaryOperation(IntegerType::Create(representation.nbits())),
        Representation_(std::move(representation))
  {}

  std::unique_ptr<Operation>
  copy() const override;

  std::string
  debug_string() const override;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] const IntegerValueRepresentation &
  Representation() const noexcept
  {
    return Representation_;
  }

  static rvsdg::Node &
  Create(rvsdg::Region & region, IntegerValueRepresentation representation)
  {
    return rvsdg::CreateOpNode<IntegerConstantOperation>(region, std::move(representation));
  }

  static rvsdg::Node &
  Create(rvsdg::Region & region, std::size_t numBits, std::int64_t value)
  {
    return Create(region, { numBits, value });
  }

private:
  IntegerValueRepresentation Representation_;
};

// FIXME: add documentation
class IntegerNegOperation final : public rvsdg::UnaryOperation
{
public:
  ~IntegerNegOperation() noexcept override;

  explicit IntegerNegOperation(const std::size_t numBits)
      : UnaryOperation(IntegerType::Create(numBits), IntegerType ::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::unop_reduction_path_t
  can_reduce_operand(const rvsdg::output * arg) const noexcept override;

  rvsdg::output *
  reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * arg) const override;
};

// FIXME: add documentation
class IntegerNotOperation final : public rvsdg::UnaryOperation
{
public:
  ~IntegerNotOperation() noexcept override;

  explicit IntegerNotOperation(const std::size_t numBits)
      : UnaryOperation(IntegerType::Create(numBits), IntegerType ::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::unop_reduction_path_t
  can_reduce_operand(const rvsdg::output * arg) const noexcept override;

  rvsdg::output *
  reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * arg) const override;
};

// FIXME: add documentation
class IntegerAddOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerAddOperation() noexcept override;

  explicit IntegerAddOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerSubOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerSubOperation() noexcept override;

  explicit IntegerSubOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerMulOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerMulOperation() noexcept override;

  explicit IntegerMulOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerSMulHOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerSMulHOperation() noexcept override;

  explicit IntegerSMulHOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerUMulHOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerUMulHOperation() noexcept override;

  explicit IntegerUMulHOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerSDivOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerSDivOperation() noexcept override;

  explicit IntegerSDivOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerUDivOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerUDivOperation() noexcept override;

  explicit IntegerUDivOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerSModOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerSModOperation() noexcept override;

  explicit IntegerSModOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerUModOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerUModOperation() noexcept override;

  explicit IntegerUModOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerAShrOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerAShrOperation() noexcept override;

  explicit IntegerAShrOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerShlOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerShlOperation() noexcept override;

  explicit IntegerShlOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerShrOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerShrOperation() noexcept override;

  explicit IntegerShrOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerAndOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerAndOperation() noexcept override;

  explicit IntegerAndOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerOrOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerOrOperation() noexcept override;

  explicit IntegerOrOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerXorOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerXorOperation() noexcept override;

  explicit IntegerXorOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerEqOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerEqOperation() noexcept override;

  explicit IntegerEqOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerNeOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerNeOperation() noexcept override;

  explicit IntegerNeOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerSgeOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerSgeOperation() noexcept override;

  explicit IntegerSgeOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerSgtOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerSgtOperation() noexcept override;

  explicit IntegerSgtOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerSleOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerSleOperation() noexcept override;

  explicit IntegerSleOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerSltOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerSltOperation() noexcept override;

  explicit IntegerSltOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerUgeOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerUgeOperation() noexcept override;

  explicit IntegerUgeOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerUgtOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerUgtOperation() noexcept override;

  explicit IntegerUgtOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerUleOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerUleOperation() noexcept override;

  explicit IntegerUleOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

// FIXME: add documentation
class IntegerUltOperation final : public rvsdg::BinaryOperation
{
public:
  ~IntegerUltOperation() noexcept override;

  explicit IntegerUltOperation(const std::size_t numBits)
      : BinaryOperation(
            { IntegerType::Create(numBits), IntegerType::Create(numBits) },
            IntegerType::Create(numBits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  rvsdg::output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  enum flags
  flags() const noexcept override;
};

}

#endif // JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
