/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_UNARY_HPP
#define JLM_RVSDG_UNARY_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>

#include <optional>

namespace jlm::rvsdg
{

typedef size_t unop_reduction_path_t;

/**
  \brief Unary operator

  Operator taking a single argument.
*/
class UnaryOperation : public SimpleOperation
{
public:
  ~UnaryOperation() noexcept override;

  UnaryOperation(
      std::shared_ptr<const jlm::rvsdg::Type> operand,
      std::shared_ptr<const jlm::rvsdg::Type> result)
      : SimpleOperation({ std::move(operand) }, { std::move(result) })
  {}

  virtual unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept = 0;

  virtual jlm::rvsdg::Output *
  reduce_operand(unop_reduction_path_t path, jlm::rvsdg::Output * arg) const = 0;
};

static const unop_reduction_path_t unop_reduction_none = 0;
/* operation is applied to constant, compute immediately */
static const unop_reduction_path_t unop_reduction_constant = 1;
/* operation does not change input operand */
static const unop_reduction_path_t unop_reduction_idempotent = 2;
/* operation is applied on inverse operation, can eliminate */
static const unop_reduction_path_t unop_reduction_inverse = 4;
/* operation "supersedes" immediately preceding operation */
static const unop_reduction_path_t unop_reduction_narrow = 5;
/* operation can be distributed into operands of preceding operation */
static const unop_reduction_path_t unop_reduction_distribute = 6;

}

#endif
