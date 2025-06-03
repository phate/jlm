/*
 * Copyright 2024 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_FUNCTIONPOINTER_HPP
#define JLM_LLVM_IR_OPERATORS_FUNCTIONPOINTER_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::llvm
{

/**
  \brief Get address of compiled function object.
  */
class FunctionToPointerOperation final : public rvsdg::UnaryOperation
{
public:
  ~FunctionToPointerOperation() noexcept override;

  FunctionToPointerOperation(std::shared_ptr<const rvsdg::FunctionType> fn);

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(rvsdg::unop_reduction_path_t path, jlm::rvsdg::Output * arg) const override;

  static std::unique_ptr<FunctionToPointerOperation>
  Create(std::shared_ptr<const rvsdg::FunctionType> fn);

  inline const std::shared_ptr<const jlm::rvsdg::FunctionType> &
  FunctionType() const noexcept
  {
    return FunctionType_;
  }

private:
  std::shared_ptr<const rvsdg::FunctionType> FunctionType_;
};

/**
  \brief Interpret pointer as callable function.
  */
class PointerToFunctionOperation final : public rvsdg::UnaryOperation
{
public:
  ~PointerToFunctionOperation() noexcept override;

  PointerToFunctionOperation(std::shared_ptr<const rvsdg::FunctionType> fn);

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::Operation>
  copy() const override;

  rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(rvsdg::unop_reduction_path_t path, jlm::rvsdg::Output * arg) const override;

  static std::unique_ptr<PointerToFunctionOperation>
  Create(std::shared_ptr<const rvsdg::FunctionType> fn);

  inline const std::shared_ptr<const rvsdg::FunctionType> &
  FunctionType() const noexcept
  {
    return FunctionType_;
  }

private:
  std::shared_ptr<const rvsdg::FunctionType> FunctionType_;
};

}

#endif // JLM_LLVM_IR_OPERATORS_FUNCTIONPOINTER_HPP
