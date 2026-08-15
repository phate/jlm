/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_BITMANIPULATIONINTRINSICOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_BITMANIPULATIONINTRINSICOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents LLVM's llvm.fshl.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-fshl-intrinsic) for more details.
 */
class FShlOperation final : public rvsdg::SimpleOperation
{
public:
  ~FShlOperation() noexcept override;

  explicit FShlOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : SimpleOperation({ type, type, type }, { type })
  {
    checkType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getType() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & operand1, const Variable & operand2, const Variable & operand3)
  {
    auto operation = std::make_unique<FShlOperation>(operand1.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand1, &operand2, &operand3 });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand1, rvsdg::Output & operand2, rvsdg::Output & operand3)
  {
    return rvsdg::CreateOpNode<FShlOperation>({ &operand1, &operand2, &operand3 }, operand1.Type());
  }

private:
  static void
  checkType(const std::shared_ptr<const rvsdg::Type> & type);
};

/**
 * Represents LLVM's llvm.ctlz.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-ctlz-intrinsic) for more details.
 */
class CtlzOperation final : public rvsdg::SimpleOperation
{
public:
  ~CtlzOperation() noexcept override;

  explicit CtlzOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : SimpleOperation({ type, rvsdg::BitType::Create(1) }, { type })
  {
    checkType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getType() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & operand, const Variable & isZeroPoison)
  {
    auto operation = std::make_unique<FShlOperation>(operand.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand, &isZeroPoison });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand, rvsdg::Output & isZeroPoison)
  {
    return rvsdg::CreateOpNode<FShlOperation>({ &operand, &isZeroPoison }, operand.Type());
  }

private:
  static void
  checkType(const std::shared_ptr<const rvsdg::Type> & type);
};

}

#endif
