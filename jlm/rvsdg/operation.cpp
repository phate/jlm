/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>

namespace jlm::rvsdg
{

Operation::~Operation() noexcept = default;

jlm::rvsdg::node_normal_form *
Operation::normal_form(Graph * graph) noexcept
{
  return graph->GetNodeNormalForm(typeid(Operation));
}

SimpleOperation::~SimpleOperation() noexcept = default;

size_t
SimpleOperation::narguments() const noexcept
{
  return operands_.size();
}

const std::shared_ptr<const rvsdg::Type> &
SimpleOperation::argument(size_t index) const noexcept
{
  JLM_ASSERT(index < narguments());
  return operands_[index];
}

size_t
SimpleOperation::nresults() const noexcept
{
  return results_.size();
}

const std::shared_ptr<const rvsdg::Type> &
SimpleOperation::result(size_t index) const noexcept
{
  JLM_ASSERT(index < nresults());
  return results_[index];
}

jlm::rvsdg::simple_normal_form *
SimpleOperation::normal_form(Graph * graph) noexcept
{
  return static_cast<simple_normal_form *>(graph->GetNodeNormalForm(typeid(SimpleOperation)));
}

bool
StructuralOperation::operator==(const Operation & other) const noexcept
{
  return typeid(*this) == typeid(other);
}

}
