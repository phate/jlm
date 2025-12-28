/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TESTOPERATIONS_HPP
#define JLM_RVSDG_TESTOPERATIONS_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/nullary.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::rvsdg
{

class TestNullaryOperation final : public NullaryOperation
{
public:
  explicit TestNullaryOperation(const std::shared_ptr<const Type> & resultType)
      : NullaryOperation(resultType)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    const auto nullaryOperation = dynamic_cast<const TestNullaryOperation *>(&other);
    return nullaryOperation && *result(0) == *nullaryOperation->result(0);
  }

  [[nodiscard]] std::string
  debug_string() const override
  {
    return "NullaryOperation";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<TestNullaryOperation>(this->result(0));
  }
};

class TestUnaryOperation final : public UnaryOperation
{
public:
  ~TestUnaryOperation() noexcept override;

  TestUnaryOperation(
      std::shared_ptr<const Type> operandType,
      std::shared_ptr<const Type> resultType) noexcept
      : UnaryOperation(std::move(operandType), std::move(resultType))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  unop_reduction_path_t
  can_reduce_operand(const Output * operand) const noexcept override;

  Output *
  reduce_operand(unop_reduction_path_t path, Output * operand) const override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static Node *
  create(
      Region *,
      std::shared_ptr<const Type> operandType,
      Output * operand,
      std::shared_ptr<const Type> resultType)
  {
    return &rvsdg::CreateOpNode<TestUnaryOperation>(
        { operand },
        std::move(operandType),
        std::move(resultType));
  }

  static Output *
  create_normalized(
      std::shared_ptr<const Type> operandType,
      Output * operand,
      std::shared_ptr<const Type> resultType)
  {
    return rvsdg::CreateOpNode<TestUnaryOperation>(
               { operand },
               std::move(operandType),
               std::move(resultType))
        .output(0);
  }
};

class TestBinaryOperation final : public BinaryOperation
{
public:
  ~TestBinaryOperation() noexcept override;

  TestBinaryOperation(
      const std::shared_ptr<const Type> & operandType,
      std::shared_ptr<const Type> resultType,
      const enum BinaryOperation::flags & flags) noexcept
      : BinaryOperation({ operandType, operandType }, std::move(resultType)),
        flags_(flags)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  binop_reduction_path_t
  can_reduce_operand_pair(const Output * op1, const Output * op2) const noexcept override;

  Output *
  reduce_operand_pair(unop_reduction_path_t path, Output * op1, Output * op2) const override;

  enum BinaryOperation::flags
  flags() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static Node *
  create(
      const std::shared_ptr<const Type> & operandType,
      std::shared_ptr<const Type> resultType,
      Output * op1,
      Output * op2)
  {
    return &rvsdg::CreateOpNode<TestBinaryOperation>(
        { op1, op2 },
        operandType,
        std::move(resultType),
        flags::none);
  }

  static Output *
  create_normalized(
      const std::shared_ptr<const Type> operandType,
      std::shared_ptr<const Type> resultType,
      Output * op1,
      Output * op2)
  {
    return rvsdg::CreateOpNode<TestBinaryOperation>(
               { op1, op2 },
               operandType,
               std::move(resultType),
               flags::none)
        .output(0);
  }

private:
  enum BinaryOperation::flags flags_;
};

}

#endif
