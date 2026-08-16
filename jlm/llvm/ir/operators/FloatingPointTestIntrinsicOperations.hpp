/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_FLOATINGPOINTTESTINTRINSICOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_FLOATINGPOINTTESTINTRINSICOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents LLVM's llvm.is.fpclass intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-is-fpclass-intrinsic) for more details.
 */
class IsFPClassOperation final : public rvsdg::SimpleOperation
{
public:
  ~IsFPClassOperation() noexcept override;

  explicit IsFPClassOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : SimpleOperation({ type, rvsdg::BitType::Create(32) }, { createResultType(type) })
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
  createTac(const Variable & operand1, const Variable & operand2)
  {
    auto operation = std::make_unique<IsFPClassOperation>(operand1.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand1, &operand2 });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IsFPClassOperation>({ &operand1, &operand2 }, operand1.Type());
  }

private:
  static void
  checkType(const std::shared_ptr<const rvsdg::Type> & type);

  static std::shared_ptr<const rvsdg::Type>
  createResultType(const std::shared_ptr<const rvsdg::Type> & type);
};

}

#endif
