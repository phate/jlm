/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>

namespace jlm::llvm
{

class IntegerBinaryOperation : public rvsdg::BinaryOperation
{
public:
  ~IntegerBinaryOperation() override;

  explicit IntegerBinaryOperation(const std::shared_ptr<const rvsdg::bittype> & type) noexcept
      : BinaryOperation({ type, type }, type)
  {}

  [[nodiscard]] const rvsdg::bittype &
  Type() const noexcept
  {
    return *util::AssertedCast<const rvsdg::bittype>(argument(0).get());
  }
};

/**
 * This operation is equivalent to LLVM's 'add' instruction.
 */
class IntegerAddOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAddOperation() noexcept override;

  explicit IntegerAddOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'sub' instruction.
 */
class IntegerSubOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSubOperation() noexcept override;

  explicit IntegerSubOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'mul' instruction.
 */
class IntegerMulOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerMulOperation() noexcept override;

  explicit IntegerMulOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'sdiv' instruction.
 */
class IntegerSDivOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSDivOperation() noexcept override;

  explicit IntegerSDivOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'udiv' instruction.
 */
class IntegerUDivOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUDivOperation() noexcept override;

  explicit IntegerUDivOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'srem' instruction.
 */
class IntegerSRemOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSRemOperation() noexcept override;

  explicit IntegerSRemOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'urem' instruction.
 */
class IntegerURemOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerURemOperation() noexcept override;

  explicit IntegerURemOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'ashr' instruction.
 */
class IntegerAShrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAShrOperation() noexcept override;

  explicit IntegerAShrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'shl' instruction.
 */
class IntegerShlOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerShlOperation() noexcept override;

  explicit IntegerShlOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'lshr' instruction.
 */
class IntegerLShrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerLShrOperation() noexcept override;

  explicit IntegerLShrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'and' instruction.
 */
class IntegerAndOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAndOperation() noexcept override;

  explicit IntegerAndOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'or' instruction.
 */
class IntegerOrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerOrOperation() noexcept override;

  explicit IntegerOrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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

/**
 * This operation is equivalent to LLVM's 'xor' instruction.
 */
class IntegerXorOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerXorOperation() noexcept override;

  explicit IntegerXorOperation(const std::size_t numBits)
      : IntegerBinaryOperation(rvsdg::bittype::Create(numBits))
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
