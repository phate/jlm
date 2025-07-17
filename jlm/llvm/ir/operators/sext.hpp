/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_SEXT_HPP
#define JLM_LLVM_IR_OPERATORS_SEXT_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::llvm
{

class SExtOperation final : public rvsdg::UnaryOperation
{
public:
  ~SExtOperation() noexcept override;

  SExtOperation(
      std::shared_ptr<const rvsdg::bittype> otype,
      std::shared_ptr<const rvsdg::bittype> rtype)
      : UnaryOperation(otype, rtype)
  {
    if (otype->nbits() >= rtype->nbits())
      throw jlm::util::error("expected operand's #bits to be smaller than results's #bits.");
  }

  inline SExtOperation(
      std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    auto ot = std::dynamic_pointer_cast<const rvsdg::bittype>(srctype);
    if (!ot)
      throw jlm::util::error("expected bits type.");

    auto rt = std::dynamic_pointer_cast<const rvsdg::bittype>(dsttype);
    if (!rt)
      throw jlm::util::error("expected bits type.");

    if (ot->nbits() >= rt->nbits())
      throw jlm::util::error("expected operand's #bits to be smaller than results' #bits.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::unop_reduction_path_t
  can_reduce_operand(const rvsdg::Output * operand) const noexcept override;

  rvsdg::Output *
  reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const override;

  inline size_t
  nsrcbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::bittype>(argument(0))->nbits();
  }

  inline size_t
  ndstbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::bittype>(result(0))->nbits();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * operand, const std::shared_ptr<const rvsdg::Type> & type)
  {
    auto ot = std::dynamic_pointer_cast<const rvsdg::bittype>(operand->Type());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    auto rt = std::dynamic_pointer_cast<const rvsdg::bittype>(type);
    if (!rt)
      throw jlm::util::error("expected bits type.");

    SExtOperation op(std::move(ot), std::move(rt));
    return ThreeAddressCode::create(op, { operand });
  }

  static rvsdg::Output *
  create(size_t ndstbits, rvsdg::Output * operand)
  {
    auto ot = std::dynamic_pointer_cast<const rvsdg::bittype>(operand->Type());
    if (!ot)
      throw util::error("expected bits type.");

    return rvsdg::CreateOpNode<SExtOperation>(
               { operand },
               std::move(ot),
               rvsdg::bittype::Create(ndstbits))
        .output(0);
  }
};

}

#endif
