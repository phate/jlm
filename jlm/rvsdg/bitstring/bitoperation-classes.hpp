/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_BITOPERATION_CLASSES_HPP
#define JLM_RVSDG_BITSTRING_BITOPERATION_CLASSES_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::rvsdg
{

/* Represents a unary operation on a bitstring of a specific width,
 * produces another bitstring of the same width. */
class bitunary_op : public jlm::rvsdg::unary_op
{
public:
  virtual ~bitunary_op() noexcept;

  inline bitunary_op(const bittype & type) noexcept
      : unary_op(type, type)
  {}

  inline const bittype &
  type() const noexcept
  {
    return *static_cast<const bittype *>(&argument(0).type());
  }

  /* reduction methods */
  virtual unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * arg) const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand(unop_reduction_path_t path, jlm::rvsdg::output * arg) const override;

  virtual bitvalue_repr
  reduce_constant(const bitvalue_repr & arg) const = 0;

  virtual std::unique_ptr<bitunary_op>
  create(size_t nbits) const = 0;
};

/* Represents a binary operation (possibly normalized n-ary if associative)
 * on a bitstring of a specific width, produces another bitstring of the
 * same width. */
class bitbinary_op : public jlm::rvsdg::binary_op
{
public:
  virtual ~bitbinary_op() noexcept;

  inline bitbinary_op(const bittype & type, size_t arity = 2) noexcept
      : binary_op(std::vector<jlm::rvsdg::port>(arity, { type }), type)
  {}

  /* reduction methods */
  virtual binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::output * arg1, const jlm::rvsdg::output * arg2)
      const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand_pair(
      binop_reduction_path_t path,
      jlm::rvsdg::output * arg1,
      jlm::rvsdg::output * arg2) const override;

  virtual bitvalue_repr
  reduce_constants(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const = 0;

  virtual std::unique_ptr<bitbinary_op>
  create(size_t nbits) const = 0;

  inline const bittype &
  type() const noexcept
  {
    return *static_cast<const bittype *>(&result(0).type());
  }
};

enum class compare_result
{
  undecidable,
  static_true,
  static_false
};

class bitcompare_op : public jlm::rvsdg::binary_op
{
public:
  virtual ~bitcompare_op() noexcept;

  inline bitcompare_op(const bittype & type) noexcept
      : binary_op({ type, type }, *bittype::Create(1))
  {}

  virtual binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::output * arg1, const jlm::rvsdg::output * arg2)
      const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand_pair(
      binop_reduction_path_t path,
      jlm::rvsdg::output * arg1,
      jlm::rvsdg::output * arg2) const override;

  virtual compare_result
  reduce_constants(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const = 0;

  virtual std::unique_ptr<bitcompare_op>
  create(size_t nbits) const = 0;

  inline const bittype &
  type() const noexcept
  {
    return *static_cast<const bittype *>(&argument(0).type());
  }
};

}

#endif
