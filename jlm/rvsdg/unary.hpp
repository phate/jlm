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

#include <optional>

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
      Graph * graph);

  virtual bool
  normalize_node(Node * node) const override;

  virtual std::vector<jlm::rvsdg::output *>
  normalized_create(
      rvsdg::Region * region,
      const SimpleOperation & op,
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
class unary_op : public SimpleOperation
{
public:
  virtual ~unary_op() noexcept;

  inline unary_op(
      std::shared_ptr<const jlm::rvsdg::Type> operand,
      std::shared_ptr<const jlm::rvsdg::Type> result)
      : SimpleOperation({ std::move(operand) }, { std::move(result) })
  {}

  virtual unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * arg) const noexcept = 0;

  virtual jlm::rvsdg::output *
  reduce_operand(unop_reduction_path_t path, jlm::rvsdg::output * arg) const = 0;

  static jlm::rvsdg::unary_normal_form *
  normal_form(Graph * graph) noexcept
  {
    return static_cast<jlm::rvsdg::unary_normal_form *>(graph->GetNodeNormalForm(typeid(unary_op)));
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

/**
 * \brief Applies the reductions implemented in the unary operations reduction functions.
 *
 * @param operation The unary operation on which the transformation is performed.
 * @param operands The single(!) operand of the unary node. It should only be a single operand.
 *
 * @return If the normalization could be applied, then the single(!) result of the unary operation
 * after the transformation. Otherwise, std::nullopt.
 *
 * \see unary_op::can_reduce_operand()
 * \see unary_op::reduce_operand()
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeUnaryOperation(const unary_op & operation, const std::vector<rvsdg::output *> & operands);

}

#endif
