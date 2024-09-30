/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>
#include <jlm/rvsdg/structural-normal-form.hpp>

namespace jlm::rvsdg
{

operation::~operation() noexcept
{}

jlm::rvsdg::node_normal_form *
operation::normal_form(jlm::rvsdg::graph * graph) noexcept
{
  return graph->node_normal_form(typeid(operation));
}

/* simple operation */

simple_op::~simple_op()
{}

size_t
simple_op::narguments() const noexcept
{
  return operands_.size();
}

const std::shared_ptr<const rvsdg::Type> &
simple_op::argument(size_t index) const noexcept
{
  JLM_ASSERT(index < narguments());
  return operands_[index];
}

size_t
simple_op::nresults() const noexcept
{
  return results_.size();
}

const std::shared_ptr<const rvsdg::Type> &
simple_op::result(size_t index) const noexcept
{
  JLM_ASSERT(index < nresults());
  return results_[index];
}

jlm::rvsdg::simple_normal_form *
simple_op::normal_form(jlm::rvsdg::graph * graph) noexcept
{
  return static_cast<jlm::rvsdg::simple_normal_form *>(graph->node_normal_form(typeid(simple_op)));
}

/* structural operation */

bool
structural_op::operator==(const operation & other) const noexcept
{
  return typeid(*this) == typeid(other);
}

jlm::rvsdg::structural_normal_form *
structural_op::normal_form(jlm::rvsdg::graph * graph) noexcept
{
  return static_cast<structural_normal_form *>(graph->node_normal_form(typeid(structural_op)));
}

}
