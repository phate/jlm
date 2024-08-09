/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/nullary.hpp>

namespace jlm::rvsdg
{

class nullary_normal_form final : public simple_normal_form
{
public:
  virtual ~nullary_normal_form() noexcept
  {}

  nullary_normal_form(
      const std::type_info & operator_class,
      jlm::rvsdg::node_normal_form * parent,
      jlm::rvsdg::graph * graph)
      : simple_normal_form(operator_class, parent, graph)
  {}
};

/* nullary operator */

nullary_op::~nullary_op() noexcept
{}

}

namespace
{

jlm::rvsdg::node_normal_form *
nullary_operation_get_default_normal_form_(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::graph * graph)
{
  return new jlm::rvsdg::nullary_normal_form(operator_class, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::rvsdg::nullary_op),
      nullary_operation_get_default_normal_form_);
}

}
