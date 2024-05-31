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

MemoryStateSplitOperation::~MemoryStateSplitOperation() noexcept = default;

bool
MemoryStateSplitOperation::operator==(const rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemoryStateSplitOperation *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
MemoryStateSplitOperation::debug_string() const
{
  return "MemoryStateSplit";
}

std::unique_ptr<rvsdg::operation>
MemoryStateSplitOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new MemoryStateSplitOperation(*this));
}

LambdaEntryMemoryStateSplitOperation::~LambdaEntryMemoryStateSplitOperation() = default;

bool
LambdaEntryMemoryStateSplitOperation::operator==(const rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const LambdaEntryMemoryStateSplitOperation *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
LambdaEntryMemoryStateSplitOperation::debug_string() const
{
  return "LambdaEntryMemoryStateSplit";
}

std::unique_ptr<jlm::rvsdg::operation>
LambdaEntryMemoryStateSplitOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new LambdaEntryMemoryStateSplitOperation(*this));
}

LambdaExitMemoryStateMergeOperation::~LambdaExitMemoryStateMergeOperation() = default;

bool
LambdaExitMemoryStateMergeOperation::operator==(const rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const LambdaExitMemoryStateMergeOperation *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
LambdaExitMemoryStateMergeOperation::debug_string() const
{
  return "LambdaExitMemoryStateMerge";
}

std::unique_ptr<rvsdg::operation>
LambdaExitMemoryStateMergeOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new LambdaExitMemoryStateMergeOperation(*this));
}

CallEntryMemoryStateMergeOperation::~CallEntryMemoryStateMergeOperation() = default;

bool
CallEntryMemoryStateMergeOperation::operator==(const rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const CallEntryMemoryStateMergeOperation *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
CallEntryMemoryStateMergeOperation::debug_string() const
{
  return "CallEntryMemoryStateMerge";
}

std::unique_ptr<rvsdg::operation>
CallEntryMemoryStateMergeOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new CallEntryMemoryStateMergeOperation(*this));
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
