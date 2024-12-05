/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NODENORMALIZATION_HPP
#define JLM_RVSDG_NODENORMALIZATION_HPP

#include <vector>

namespace jlm::rvsdg
{

class node;
class output;

/**
 * \brief Normalizes the operands of an operation.
 *
 * The usage expectation is that IsApplicable() is invoked to determine whether operands can be
 * normalized BEFORE ApplyNormalization() is invoked.
 *
 * @tparam TOperation The operation for which to normalize the operands.
 */
template<class TOperation>
class NodeNormalization
{
  static_assert(
      std::is_base_of<operation, TOperation>::value,
      "Template parameter TOperation must be derived from jlm::rvsdg::operation.");

public:
  virtual ~NodeNormalization() noexcept = default;

  NodeNormalization() = default;

  NodeNormalization(const NodeNormalization &) = delete;

  NodeNormalization(NodeNormalization &&) = delete;

  NodeNormalization &
  operator=(const NodeNormalization &) = delete;

  NodeNormalization &
  operator=(NodeNormalization &&) = delete;

  /**
   * Determines whether the operands can be normalized.
   *
   * \note While the method can modify internal state, it is expected to be re-entrant.
   *
   * @param operation The operation for which to normalize the operands.
   * @param operands The operands corresponding to the operation.
   *
   * @return True, if the operands can be normalized, otherwise false.
   */
  virtual bool
  IsApplicable(const TOperation & operation, const std::vector<output *> & operands) = 0;

  /**
   * \brief Performs the normalization of the operands.
   *
   * @param operation The operation for which to normalize the operands.
   * @param operands The operands corresponding to the operation.
   *
   * @return The normalized operands.
   */
  virtual std::vector<output *>
  ApplyNormalization(const TOperation & operation, const std::vector<output *> & operands) = 0;
};

}

#endif
