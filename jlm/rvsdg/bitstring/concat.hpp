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

class bitconcat_op final : public jlm::rvsdg::binary_op
{
public:
  virtual ~bitconcat_op() noexcept;

  explicit inline bitconcat_op(const std::vector<std::shared_ptr<const bittype>> types)
      : binary_op({ types.begin(), types.end() }, aggregate_arguments(types))
  {}

  virtual bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override;

  virtual binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::output * arg1, const jlm::rvsdg::output * arg2)
      const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand_pair(
      binop_reduction_path_t path,
      jlm::rvsdg::output * arg1,
      jlm::rvsdg::output * arg2) const override;

  virtual enum jlm::rvsdg::binary_op::flags
  flags() const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

private:
  static std::shared_ptr<const bittype>
  aggregate_arguments(const std::vector<std::shared_ptr<const bittype>> & types) noexcept;
};

jlm::rvsdg::output *
bitconcat(const std::vector<jlm::rvsdg::output *> & operands);

}

#endif
