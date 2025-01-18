/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>

namespace jlm::rvsdg
{

Operation::~Operation() noexcept = default;

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

bool
StructuralOperation::operator==(const Operation & other) const noexcept
{
  return typeid(*this) == typeid(other);
}

}
