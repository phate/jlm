/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_SLICE_HPP
#define JLM_RVSDG_BITSTRING_SLICE_HPP

#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::rvsdg
{

class bitslice_op : public jlm::rvsdg::unary_op
{
public:
  virtual ~bitslice_op() noexcept;

  inline bitslice_op(const bittype & argument, size_t low, size_t high) noexcept
      : unary_op(argument, bittype(high - low)),
        low_(low)
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * arg) const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand(unop_reduction_path_t path, jlm::rvsdg::output * arg) const override;

  inline size_t
  low() const noexcept
  {
    return low_;
  }

  inline size_t
  high() const noexcept
  {
    return low_ + static_cast<const bittype *>(&result(0).type())->nbits();
  }

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  inline const type &
  argument_type() const noexcept
  {
    return *static_cast<const bittype *>(&argument(0).type());
  }

private:
  size_t low_;
};

/**
  \brief Create bitslice
  \param operand Input value
  \param low Low bit
  \param high High bit
  \returns Bitstring value representing slice

  Convenience function that either creates a new slice or
  returns the output handle of an existing slice.
*/
jlm::rvsdg::output *
bitslice(jlm::rvsdg::output * operand, size_t low, size_t high);

}

#endif
