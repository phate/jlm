/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_UNARY_HPP
#define JLM_RVSDG_UNARY_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>
#include <jlm/util/common.hpp>

namespace jlm::rvsdg
{

typedef size_t unop_reduction_path_t;

class unary_normal_form final : public simple_normal_form
{
public:
  virtual ~unary_normal_form() noexcept;

  unary_normal_form(
      const std::type_info & operator_class,
      jlm::rvsdg::node_normal_form * parent,
      jlm::rvsdg::graph * graph);

  virtual bool
  normalize_node(jlm::rvsdg::node * node) const override;

  virtual std::vector<jlm::rvsdg::output *>
  normalized_create(
      jlm::rvsdg::region * region,
      const jlm::rvsdg::simple_op & op,
      const std::vector<jlm::rvsdg::output *> & arguments) const override;

  virtual void
  set_reducible(bool enable);

  inline bool
  get_reducible() const noexcept
  {
    return enable_reducible_;
  }

private:
  bool enable_reducible_;
};

/**
  \brief Unary operator

  Operator taking a single argument.
*/
class unary_op : public simple_op
{
public:
  virtual ~unary_op() noexcept;

  inline unary_op(const jlm::rvsdg::port & operand, const jlm::rvsdg::port & result)
      : simple_op({ operand }, { result })
  {}

  virtual unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * arg) const noexcept = 0;

  virtual jlm::rvsdg::output *
  reduce_operand(unop_reduction_path_t path, jlm::rvsdg::output * arg) const = 0;

  static jlm::rvsdg::unary_normal_form *
  normal_form(jlm::rvsdg::graph * graph) noexcept
  {
    return static_cast<jlm::rvsdg::unary_normal_form *>(graph->node_normal_form(typeid(unary_op)));
  }
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
