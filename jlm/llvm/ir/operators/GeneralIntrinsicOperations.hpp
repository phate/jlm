/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_GENERALINTRINSICOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_GENERALINTRINSICOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents LLVM's llvm.is.constant.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-is-constant-intrinsic) for more details.
 */
class IsConstantOperation final : public rvsdg::UnaryOperation
{
public:
  ~IsConstantOperation() noexcept override;

  explicit IsConstantOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : UnaryOperation(type, rvsdg::BitType::Create(1))
  {}

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
  createTac(const Variable & operand)
  {
    auto operation = std::make_unique<IsConstantOperation>(operand.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<IsConstantOperation>({ &operand }, operand.Type());
  }
};

/**
 * Represents LLVM's llvm.ptrmask intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-ptrmask-intrinsic) for more details.
 */
class PtrMaskOperation final : public rvsdg::SimpleOperation
{
public:
  ~PtrMaskOperation() noexcept override;

  explicit PtrMaskOperation(
      const std::shared_ptr<const rvsdg::Type> & ptrOperandType,
      const std::shared_ptr<const rvsdg::Type> & maskOperandType)
      : SimpleOperation({ ptrOperandType, maskOperandType }, { ptrOperandType })
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getPtrOperandType() const noexcept
  {
    return argument(0);
  }

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getMaskOperandType() const noexcept
  {
    return argument(1);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & ptrOperand, const Variable & maskOperand)
  {
    auto operation = std::make_unique<PtrMaskOperation>(ptrOperand.Type(), maskOperand.Type());
    return ThreeAddressCode::create(std::move(operation), { &ptrOperand, &maskOperand });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & ptrOperand, rvsdg::Output & maskOperand)
  {
    return rvsdg::CreateOpNode<PtrMaskOperation>(
        { &ptrOperand, &maskOperand },
        ptrOperand.Type(),
        maskOperand.Type());
  }

private:
  static void
  checkOperandTypes(
      const std::shared_ptr<const rvsdg::Type> & ptrType,
      const std::shared_ptr<const rvsdg::Type> & maskType);
};

}

#endif
