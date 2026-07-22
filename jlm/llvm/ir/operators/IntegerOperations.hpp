/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/nullary.hpp>

namespace jlm::llvm
{

// FIXME: Implement our own value representation instead of re-using the bitstring value
// representation
using IntegerValueRepresentation = rvsdg::BitValueRepresentation;

/**
 * Represents an LLVM integer constant
 */
class IntegerConstantOperation final : public rvsdg::NullaryOperation
{
public:
  ~IntegerConstantOperation() override;

  explicit IntegerConstantOperation(IntegerValueRepresentation representation)
      : NullaryOperation(rvsdg::BitType::Create(representation.nbits())),
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

/**
 * Represents an LLVM integer binary operation
 */
class IntegerBinaryOperation : public rvsdg::BinaryOperation
{
public:
  ~IntegerBinaryOperation() noexcept override;

  IntegerBinaryOperation(
      const std::size_t numArgumentBits,
      const std::size_t numResultBits) noexcept
      : BinaryOperation(
            { rvsdg::BitType::Create(numArgumentBits), rvsdg::BitType::Create(numArgumentBits) },
            rvsdg::BitType::Create(numResultBits))
  {}

  [[nodiscard]] const rvsdg::BitType &
  Type() const noexcept
  {
    return *util::assertedCast<const rvsdg::BitType>(argument(0).get());
  }
};

/**
 * This operation is equivalent to LLVM's 'add' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#add-instruction) for more
 * details.
 */
class IntegerAddOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAddOperation() noexcept override;

  explicit IntegerAddOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerAddOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'sub' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#sub-instruction) for more
 * details.
 */
class IntegerSubOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSubOperation() noexcept override;

  explicit IntegerSubOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSubOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'mul' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#mul-instruction) for more
 * details.
 */
class IntegerMulOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerMulOperation() noexcept override;

  explicit IntegerMulOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerMulOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'sdiv' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#sdiv-instruction) for
 * more details.
 */
class IntegerSDivOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSDivOperation() noexcept override;

  explicit IntegerSDivOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSDivOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'udiv' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#udiv-instruction) for
 * more details.
 */
class IntegerUDivOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUDivOperation() noexcept override;

  explicit IntegerUDivOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUDivOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'srem' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#srem-instruction) for
 * more details.
 */
class IntegerSRemOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSRemOperation() noexcept override;

  explicit IntegerSRemOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSRemOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'urem' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#urem-instruction) for
 * more details.
 */
class IntegerURemOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerURemOperation() noexcept override;

  explicit IntegerURemOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerURemOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'ashr' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#ashr-instruction) for
 * more details.
 */
class IntegerAShrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAShrOperation() noexcept override;

  explicit IntegerAShrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerAShrOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'shl' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#shl-instruction) for more
 * details.
 */
class IntegerShlOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerShlOperation() noexcept override;

  explicit IntegerShlOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerShlOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'lshr' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#lshr-instruction) for
 * more details.
 */
class IntegerLShrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerLShrOperation() noexcept override;

  explicit IntegerLShrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerLShrOperation>({ &operand1, &operand2 }, numBits);
  }
};

/**
 * This operation is equivalent to LLVM's 'and' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#and-instruction) for more
 * details.
 */
class IntegerAndOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAndOperation() noexcept override;

  explicit IntegerAndOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerAndOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerAndOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerAndOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerAndOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerAndOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'or' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#or-instruction) for more
 * details.
 */
class IntegerOrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerOrOperation() noexcept override;

  explicit IntegerOrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerOrOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerOrOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerOrOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerOrOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerOrOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'xor' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#xor-instruction) for more
 * details.
 */
class IntegerXorOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerXorOperation() noexcept override;

  explicit IntegerXorOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerXorOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerXorOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerXorOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerXorOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerXorOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'eq' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerEqOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerEqOperation() noexcept override;

  explicit IntegerEqOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerEqOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerEqOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerEqOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerEqOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerEqOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'ne' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerNeOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerNeOperation() noexcept override;

  explicit IntegerNeOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerNeOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerNeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerNeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerNeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerNeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'sge' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerSgeOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSgeOperation() noexcept override;

  explicit IntegerSgeOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSgeOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerSgeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSgeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSgeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSgeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'sgt' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerSgtOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSgtOperation() noexcept override;

  explicit IntegerSgtOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSgtOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerSgtOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSgtOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSgtOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSgtOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'sle' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerSleOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSleOperation() noexcept override;

  explicit IntegerSleOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSleOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerSleOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSleOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSleOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSleOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'slt' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerSltOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSltOperation() noexcept override;

  explicit IntegerSltOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSltOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerSltOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSltOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSltOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSltOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'uge' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerUgeOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUgeOperation() noexcept override;

  explicit IntegerUgeOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUgeOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerUgeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUgeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUgeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUgeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'ugt' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerUgtOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUgtOperation() noexcept override;

  explicit IntegerUgtOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUgtOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerUgtOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUgtOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUgtOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUgtOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'ule' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerUleOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUleOperation() noexcept override;

  explicit IntegerUleOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUleOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerUleOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUleOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUleOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUleOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'ult' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerUltOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUltOperation() noexcept override;

  explicit IntegerUltOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUltOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerUltOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUltOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUltOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUltOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

}

#endif // JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
