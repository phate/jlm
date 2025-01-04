/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::rvsdg
{

/* unary normal form */

unary_normal_form::~unary_normal_form() noexcept
{}

unary_normal_form::unary_normal_form(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    Graph * graph)
    : simple_normal_form(operator_class, parent, graph),
      enable_reducible_(true)
{
  if (auto p = dynamic_cast<unary_normal_form *>(parent))
  {
    enable_reducible_ = p->enable_reducible_;
  }
}

bool
unary_normal_form::normalize_node(Node * node) const
{
  if (!get_mutable())
  {
    return true;
  }

  const auto & op = static_cast<const unary_op &>(node->GetOperation());

  if (get_reducible())
  {
    auto tmp = node->input(0)->origin();
    unop_reduction_path_t reduction = op.can_reduce_operand(tmp);
    if (reduction != unop_reduction_none)
    {
      divert_users(node, { op.reduce_operand(reduction, tmp) });
      remove(node);
      return false;
    }
  }

  return simple_normal_form::normalize_node(node);
}

std::vector<jlm::rvsdg::output *>
unary_normal_form::normalized_create(
    rvsdg::Region * region,
    const SimpleOperation & op,
    const std::vector<jlm::rvsdg::output *> & arguments) const
{
  JLM_ASSERT(arguments.size() == 1);

  if (get_mutable() && get_reducible())
  {
    const auto & un_op = static_cast<const jlm::rvsdg::unary_op &>(op);

    unop_reduction_path_t reduction = un_op.can_reduce_operand(arguments[0]);
    if (reduction != unop_reduction_none)
    {
      return { un_op.reduce_operand(reduction, arguments[0]) };
    }
  }

  return simple_normal_form::normalized_create(region, op, arguments);
}

void
unary_normal_form::set_reducible(bool enable)
{
  if (get_reducible() == enable)
  {
    return;
  }

  children_set<unary_normal_form, &unary_normal_form::set_reducible>(enable);

  enable_reducible_ = enable;
  if (get_mutable() && enable)
    graph()->MarkDenormalized();
}

/* unary operator */

unary_op::~unary_op() noexcept
{}

std::optional<std::vector<rvsdg::output *>>
NormalizeUnaryOperation(const unary_op & operation, const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  auto & operand = *operands[0];

  if (const auto reduction = operation.can_reduce_operand(&operand);
      reduction != unop_reduction_none)
  {
    return { { operation.reduce_operand(reduction, &operand) } };
  }

  return std::nullopt;
}

}

jlm::rvsdg::node_normal_form *
unary_operation_get_default_normal_form_(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::Graph * graph)
{
  jlm::rvsdg::node_normal_form * nf =
      new jlm::rvsdg::unary_normal_form(operator_class, parent, graph);

  return nf;
}

static void __attribute__((constructor))
register_node_normal_form()
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::rvsdg::unary_op),
      unary_operation_get_default_normal_form_);
}
