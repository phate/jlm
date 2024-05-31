/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>

namespace jlm::llvm
{

MemoryStateMergeOperation::~MemoryStateMergeOperation() noexcept = default;

bool
MemoryStateMergeOperation::operator==(const rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemoryStateMergeOperation *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
MemoryStateMergeOperation::debug_string() const
{
  return "MemoryStateMerge";
}

std::unique_ptr<rvsdg::operation>
MemoryStateMergeOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new MemoryStateMergeOperation(*this));
}

/* MemStateSplit operator */

MemStateSplitOperator::~MemStateSplitOperator()
{}

bool
MemStateSplitOperator::operator==(const rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemStateSplitOperator *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
MemStateSplitOperator::debug_string() const
{
  return "MemStateSplit";
}

std::unique_ptr<rvsdg::operation>
MemStateSplitOperator::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new MemStateSplitOperator(*this));
}

/* LambdaEntryMemStateOperator class */

LambdaEntryMemStateOperator::~LambdaEntryMemStateOperator() = default;

bool
LambdaEntryMemStateOperator::operator==(const jlm::rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const LambdaEntryMemStateOperator *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
LambdaEntryMemStateOperator::debug_string() const
{
  return "LambdaEntryMemState";
}

std::unique_ptr<jlm::rvsdg::operation>
LambdaEntryMemStateOperator::copy() const
{
  return std::unique_ptr<jlm::rvsdg::operation>(new LambdaEntryMemStateOperator(*this));
}

/* LambdaExitMemStateOperator class */

LambdaExitMemStateOperator::~LambdaExitMemStateOperator() = default;

bool
LambdaExitMemStateOperator::operator==(const jlm::rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const LambdaExitMemStateOperator *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
LambdaExitMemStateOperator::debug_string() const
{
  return "LambdaExitMemState";
}

std::unique_ptr<jlm::rvsdg::operation>
LambdaExitMemStateOperator::copy() const
{
  return std::unique_ptr<jlm::rvsdg::operation>(new LambdaExitMemStateOperator(*this));
}

/* CallEntryMemStateOperator class */

CallEntryMemStateOperator::~CallEntryMemStateOperator() = default;

bool
CallEntryMemStateOperator::operator==(const jlm::rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const CallEntryMemStateOperator *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
CallEntryMemStateOperator::debug_string() const
{
  return "CallEntryMemState";
}

std::unique_ptr<jlm::rvsdg::operation>
CallEntryMemStateOperator::copy() const
{
  return std::unique_ptr<jlm::rvsdg::operation>(new CallEntryMemStateOperator(*this));
}

/* CallExitMemStateOperator class */

CallExitMemStateOperator::~CallExitMemStateOperator() = default;

bool
CallExitMemStateOperator::operator==(const jlm::rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const CallExitMemStateOperator *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
CallExitMemStateOperator::debug_string() const
{
  return "CallExitMemState";
}

std::unique_ptr<jlm::rvsdg::operation>
CallExitMemStateOperator::copy() const
{
  return std::unique_ptr<jlm::rvsdg::operation>(new CallExitMemStateOperator(*this));
}

}
