/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>

static jlm::rvsdg::Node *
node_cse(
    jlm::rvsdg::Region * region,
    const jlm::rvsdg::Operation & op,
    const std::vector<jlm::rvsdg::output *> & arguments)
{
  auto cse_test = [&](const jlm::rvsdg::Node * node)
  {
    return node->GetOperation() == op && arguments == operands(node);
  };

  if (!arguments.empty())
  {
    for (const auto & user : *arguments[0])
    {
      if (!jlm::rvsdg::is<jlm::rvsdg::node_input>(*user))
        continue;

      auto node = static_cast<jlm::rvsdg::node_input *>(user)->node();
      if (cse_test(node))
        return node;
    }
  }
  else
  {
    for (auto & node : region->TopNodes())
    {
      if (cse_test(&node))
        return &node;
    }
  }

  return nullptr;
}

namespace jlm::rvsdg
{

simple_normal_form::~simple_normal_form() noexcept
{}

simple_normal_form::simple_normal_form(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    Graph * graph) noexcept
    : node_normal_form(operator_class, parent, graph),
      enable_cse_(true)
{
  if (auto p = dynamic_cast<simple_normal_form *>(parent))
    enable_cse_ = p->get_cse();
}

bool
simple_normal_form::normalize_node(Node * node) const
{
  if (!get_mutable())
    return true;

  if (get_cse())
  {
    auto new_node = node_cse(node->region(), node->GetOperation(), operands(node));
    JLM_ASSERT(new_node);
    if (new_node != node)
    {
      divert_users(node, outputs(new_node));
      remove(node);
      return false;
    }
  }

  return true;
}

std::vector<jlm::rvsdg::output *>
simple_normal_form::normalized_create(
    rvsdg::Region * region,
    const SimpleOperation & op,
    const std::vector<jlm::rvsdg::output *> & arguments) const
{
  Node * node = nullptr;
  if (get_mutable() && get_cse())
    node = node_cse(region, op, arguments);
  if (!node)
    node = simple_node::create(region, op, arguments);

  return outputs(node);
}

void
simple_normal_form::set_cse(bool enable)
{
  if (enable == enable_cse_)
    return;

  enable_cse_ = enable;
  children_set<simple_normal_form, &simple_normal_form::set_cse>(enable);

  if (get_mutable() && enable)
    graph()->mark_denormalized();
}

}

static jlm::rvsdg::node_normal_form *
get_default_normal_form(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::Graph * graph)
{
  return new jlm::rvsdg::simple_normal_form(operator_class, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::rvsdg::SimpleOperation),
      get_default_normal_form);
}
