/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_SPECIALIZEDARITHMETICINTRINSICOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_SPECIALIZEDARITHMETICINTRINSICOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents LLVM's llvm.fmuladd.* intrinsic
 *
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#llvm-fmuladd-intrinsic)
 * for more details.
 */
class FMulAddIntrinsicOperation final : public rvsdg::SimpleOperation
{
public:
  ~FMulAddIntrinsicOperation() noexcept override;

  explicit FMulAddIntrinsicOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : SimpleOperation({ type, type, type }, { type })
  {
    CheckType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static rvsdg::SimpleNode &
  CreateNode(rvsdg::Output & multiplier, rvsdg::Output & multiplicand, rvsdg::Output & summand)
  {
    return rvsdg::CreateOpNode<FMulAddIntrinsicOperation>(
        { &multiplier, &multiplicand, &summand },
        multiplier.Type());
  }

  static std::unique_ptr<ThreeAddressCode>
  CreateTac(const Variable & multiplier, const Variable & multiplicand, const Variable & summand)
  {
    const FMulAddIntrinsicOperation operation(multiplier.Type());
    return ThreeAddressCode::create(operation, { &multiplier, &multiplicand, &summand });
  }

private:
  static void
  CheckType(const std::shared_ptr<const rvsdg::Type> & type);
};

}

#endif // JLM_LLVM_IR_OPERATORS_SPECIALIZEDARITHMETICINTRINSICOPERATIONS_HPP
