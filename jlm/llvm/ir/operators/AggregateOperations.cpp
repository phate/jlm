/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/AggregateOperations.hpp>

namespace jlm::llvm
{

ExtractValueOperation::~ExtractValueOperation() noexcept = default;

bool
ExtractValueOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const ExtractValueOperation *>(&other);
  return op && op->indices_ == indices_ && op->type() == type();
}

std::string
ExtractValueOperation::debug_string() const
{
  return "ExtractValue";
}

std::unique_ptr<rvsdg::Operation>
ExtractValueOperation::copy() const
{
  return std::make_unique<ExtractValueOperation>(*this);
}

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
