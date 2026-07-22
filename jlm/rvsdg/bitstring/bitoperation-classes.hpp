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
class BitUnaryOperation : public UnaryOperation
{
public:
  ~BitUnaryOperation() noexcept override;

  explicit BitUnaryOperation(const std::shared_ptr<const BitType> & type) noexcept
      : UnaryOperation(type, type)
  {}

  inline const BitType &
  type() const noexcept
  {
    return *std::static_pointer_cast<const BitType>(argument(0));
  }

  virtual BitValueRepresentation
  reduce_constant(const BitValueRepresentation & arg) const = 0;

  virtual std::unique_ptr<BitUnaryOperation>
  create(size_t nbits) const = 0;

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref BitUnaryOperation on which the transformation is performed.
   * @param operands The operands of the \ref BitUnaryOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref BitUnaryOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<Output *>>
  foldConstant(const BitUnaryOperation & operation, const std::vector<Output *> & operands);
};

/* Represents a binary operation (possibly normalized n-ary if associative)
 * on a bitstring of a specific width, produces another bitstring of the
 * same width. */
class BitBinaryOperation : public BinaryOperation
{
public:
  ~BitBinaryOperation() noexcept override;

  explicit BitBinaryOperation(const std::shared_ptr<const BitType> type, size_t arity = 2) noexcept
      : BinaryOperation({ arity, type }, type)
  {}

  /* reduction methods */
  binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::Output * arg1, const jlm::rvsdg::Output * arg2)
      const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand_pair(
      binop_reduction_path_t path,
      jlm::rvsdg::Output * arg1,
      jlm::rvsdg::Output * arg2) const override;

  virtual BitValueRepresentation
  reduce_constants(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2)
      const = 0;

  virtual std::unique_ptr<BitBinaryOperation>
  create(size_t nbits) const = 0;

  inline const BitType &
  type() const noexcept
  {
    return *std::static_pointer_cast<const BitType>(result(0));
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref BitBinaryOperation on which the transformation is performed.
   * @param operands The operands of the \ref BitBinaryOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref BitBinaryOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<Output *>>
  foldConstants(const BitBinaryOperation & operation, const std::vector<Output *> & operands);
};

enum class compare_result
{
  undecidable,
  static_true,
  static_false
};

class BitCompareOperation : public BinaryOperation
{
public:
  ~BitCompareOperation() noexcept override;

  explicit BitCompareOperation(std::shared_ptr<const BitType> type) noexcept
      : BinaryOperation({ type, type }, BitType::Create(1))
  {}

  binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::Output * arg1, const jlm::rvsdg::Output * arg2)
      const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand_pair(
      binop_reduction_path_t path,
      jlm::rvsdg::Output * arg1,
      jlm::rvsdg::Output * arg2) const override;

  virtual compare_result
  reduce_constants(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2)
      const = 0;

  virtual std::unique_ptr<BitCompareOperation>
  create(size_t nbits) const = 0;

  inline const BitType &
  type() const noexcept
  {
    return *std::static_pointer_cast<const BitType>(argument(0));
  }
};

}

#endif
