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

/* sext operator */

class sext_op final : public rvsdg::unary_op
{
public:
  virtual ~sext_op();

  inline sext_op(
      std::shared_ptr<const rvsdg::bittype> otype,
      std::shared_ptr<const rvsdg::bittype> rtype)
      : unary_op(otype, rtype)
  {
    if (otype->nbits() >= rtype->nbits())
      throw jlm::util::error("expected operand's #bits to be smaller than results's #bits.");
  }

  inline sext_op(
      std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype)
      : unary_op(srctype, dsttype)
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

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<rvsdg::operation>
  copy() const override;

  virtual rvsdg::unop_reduction_path_t
  can_reduce_operand(const rvsdg::output * operand) const noexcept override;

  virtual rvsdg::output *
  reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const override;

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

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const std::shared_ptr<const rvsdg::Type> & type)
  {
    auto ot = std::dynamic_pointer_cast<const rvsdg::bittype>(operand->Type());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    auto rt = std::dynamic_pointer_cast<const rvsdg::bittype>(type);
    if (!rt)
      throw jlm::util::error("expected bits type.");

    sext_op op(std::move(ot), std::move(rt));
    return tac::create(op, { operand });
  }

  static rvsdg::output *
  create(size_t ndstbits, rvsdg::output * operand)
  {
    auto ot = std::dynamic_pointer_cast<const rvsdg::bittype>(operand->Type());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    sext_op op(std::move(ot), rvsdg::bittype::Create(ndstbits));
    return rvsdg::simple_node::create_normalized(operand->region(), op, { operand })[0];
  }
};

}

#endif
