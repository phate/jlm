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

class BitSliceOperation final : public UnaryOperation
{
public:
  ~BitSliceOperation() noexcept override;

  BitSliceOperation(
      const std::shared_ptr<const BitType> & argument,
      size_t low,
      size_t high) noexcept
      : UnaryOperation(argument, BitType::Create(high - low)),
        low_(low)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(unop_reduction_path_t path, jlm::rvsdg::Output * arg) const override;

  inline size_t
  low() const noexcept
  {
    return low_;
  }

  inline size_t
  high() const noexcept
  {
    return low_ + std::static_pointer_cast<const BitType>(result(0))->nbits();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline const Type &
  argument_type() const noexcept
  {
    return *std::static_pointer_cast<const BitType>(argument(0));
  }

  /**
   * Removes the \ref BitSliceOperation if its slicing boundaries align with the size of the
   * operand, i.e., low == 0 and high = numBits.
   *
   * @param operation The \ref BitSliceOperation on which the transformation is performed.
   * @param operands The operands of the \ref BitSliceOperation node.
   *
   * @return If the normalization could be applied, then the operand of the \ref BitSliceOperation
   * node. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<Output *>>
  normalizeIdempotent(const BitSliceOperation & operation, const std::vector<Output *> & operands);

  /**
   * Distribute a \ref BitSliceOperation node over a \ref BitConcatOperation node:
   *
   * c = BitConcatOperation x1[8] x2[8]
   * s = BitSliceOperation[4:12] c
   * =>
   * s1 = BitSliceOperation[4:8] x1
   * s2 = BitSliceOperation[0:4] x2
   * s = BitConcatOperation s1 s2
   *
   * @param operation The \ref BitSliceOperation on which the transformation is performed.
   * @param operands The operands of the \ref BitSliceOperation node.
   *
   * @return If the distribution could be applied, then the results of the distribution. Otherwise,
   * std::nullopt.
   */
  static std::optional<std::vector<Output *>>
  distributeSlice(const BitSliceOperation & operation, const std::vector<Output *> & operands);

  /**
   * Performs constant folding by statically evaluating the constant operand and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref BitSliceOperation on which the transformation is performed.
   * @param operands The operand of the \ref BitSliceOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref BitSliceOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<Output *>>
  foldConstant(const BitSliceOperation & operation, const std::vector<Output *> & operands);

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
jlm::rvsdg::Output *
bitslice(jlm::rvsdg::Output * operand, size_t low, size_t high);

}

#endif
