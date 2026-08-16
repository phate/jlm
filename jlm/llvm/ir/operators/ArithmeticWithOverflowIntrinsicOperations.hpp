/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_ARITHMETICWITHOVERFLOWINTRINSICOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_ARITHMETICWITHOVERFLOWINTRINSICOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents LLVM's llvm.sadd.with.overflow.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-sadd-with-overflow-intrinsics) for more details.
 */
class SAddWithOverflowOperation final : public rvsdg::SimpleOperation
{
public:
  ~SAddWithOverflowOperation() noexcept override;

  explicit SAddWithOverflowOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : SimpleOperation(
            { type, type },
            { StructType::CreateLiteral({ type, rvsdg::BitType::Create(1) }, false) })
  {
    checkOperandType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getOperandType() const noexcept
  {
    return argument(0);
  }

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getResultType() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & operand1, const Variable & operand2)
  {
    auto operation = std::make_unique<SAddWithOverflowOperation>(operand1.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand1, &operand2 });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<SAddWithOverflowOperation>(
        { &operand1, &operand2 },
        operand1.Type());
  }

private:
  static void
  checkOperandType(const std::shared_ptr<const rvsdg::Type> & type);
};

}

#endif
