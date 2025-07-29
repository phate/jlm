/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_CONCAT_HPP
#define JLM_RVSDG_BITSTRING_CONCAT_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/node.hpp>

#include <vector>

namespace jlm::rvsdg
{

class BitConcatOperation final : public BinaryOperation
{
public:
  ~BitConcatOperation() noexcept override;

  explicit BitConcatOperation(const std::vector<std::shared_ptr<const bittype>> types)
      : BinaryOperation({ types.begin(), types.end() }, aggregate_arguments(types))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::Output * arg1, const jlm::rvsdg::Output * arg2)
      const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand_pair(
      binop_reduction_path_t path,
      jlm::rvsdg::Output * arg1,
      jlm::rvsdg::Output * arg2) const override;

  enum BinaryOperation::flags
  flags() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

private:
  static std::shared_ptr<const bittype>
  aggregate_arguments(const std::vector<std::shared_ptr<const bittype>> & types) noexcept;
};

jlm::rvsdg::Output *
bitconcat(const std::vector<jlm::rvsdg::Output *> & operands);

std::optional<std::vector<rvsdg::Output *>>
FlattenBitConcatOperation(
    const BitConcatOperation & operation,
    const std::vector<rvsdg::Output *> & operands);

}

#endif
