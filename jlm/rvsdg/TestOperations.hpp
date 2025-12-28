/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TESTOPERATIONS_HPP
#define JLM_RVSDG_TESTOPERATIONS_HPP

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

}

#endif
