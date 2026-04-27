/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/AggregateOperations.hpp>

namespace jlm::llvm
{

InsertValueOperation::~InsertValueOperation() noexcept = default;

std::string
InsertValueOperation::debug_string() const
{
  return "InsertValue";
}

std::unique_ptr<rvsdg::Operation>
InsertValueOperation::copy() const
{
  return create(getAggregateType(), getValueType(), getIndices());
}

bool
InsertValueOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const InsertValueOperation *>(&other);
  return operation && operation->getAggregateType() == getAggregateType()
      && operation->getValueType() == getValueType() && operation->getIndices() == getIndices();
}

}
