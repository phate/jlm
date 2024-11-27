/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/structural-normal-form.hpp>

namespace jlm::rvsdg
{

structural_normal_form::~structural_normal_form() noexcept
{}

structural_normal_form::structural_normal_form(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    Graph * graph) noexcept
    : node_normal_form(operator_class, parent, graph)
{}

}

static jlm::rvsdg::node_normal_form *
get_default_normal_form(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::Graph * graph)
{
  return new jlm::rvsdg::structural_normal_form(operator_class, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::rvsdg::StructuralOperation),
      get_default_normal_form);
}
