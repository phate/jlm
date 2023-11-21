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

  inline sext_op(const rvsdg::bittype & otype, const rvsdg::bittype & rtype)
      : unary_op({ otype }, { rtype })
  {
    if (otype.nbits() >= rtype.nbits())
      throw jlm::util::error("expected operand's #bits to be smaller than results's #bits.");
  }

  inline sext_op(std::unique_ptr<rvsdg::type> srctype, std::unique_ptr<rvsdg::type> dsttype)
      : unary_op(*srctype, *dsttype)
  {
    auto ot = dynamic_cast<const rvsdg::bittype *>(srctype.get());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    auto rt = dynamic_cast<const rvsdg::bittype *>(dsttype.get());
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
    return static_cast<const rvsdg::bittype *>(&argument(0).type())->nbits();
  }

  inline size_t
  ndstbits() const noexcept
  {
    return static_cast<const rvsdg::bittype *>(&result(0).type())->nbits();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const rvsdg::type & type)
  {
    auto ot = dynamic_cast<const rvsdg::bittype *>(&operand->type());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    auto rt = dynamic_cast<const rvsdg::bittype *>(&type);
    if (!rt)
      throw jlm::util::error("expected bits type.");

    sext_op op(*ot, *rt);
    return tac::create(op, { operand });
  }

  static rvsdg::output *
  create(size_t ndstbits, rvsdg::output * operand)
  {
    auto ot = dynamic_cast<const rvsdg::bittype *>(&operand->type());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    sext_op op(*ot, rvsdg::bittype(ndstbits));
    return rvsdg::simple_node::create_normalized(operand->region(), op, { operand })[0];
  }
};

}

#endif
