/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>

namespace jlm::llvm
{

MemoryStateMergeOperation::~MemoryStateMergeOperation() noexcept = default;

bool
MemoryStateMergeOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemoryStateMergeOperation *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
MemoryStateMergeOperation::debug_string() const
{
  return "MemoryStateMerge";
}

std::unique_ptr<rvsdg::Operation>
MemoryStateMergeOperation::copy() const
{
  return std::make_unique<MemoryStateMergeOperation>(*this);
}

MemoryStateSplitOperation::~MemoryStateSplitOperation() noexcept = default;

bool
MemoryStateSplitOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemoryStateSplitOperation *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
MemoryStateSplitOperation::debug_string() const
{
  return "MemoryStateSplit";
}

std::unique_ptr<rvsdg::Operation>
MemoryStateSplitOperation::copy() const
{
  return std::make_unique<MemoryStateSplitOperation>(*this);
}

LambdaEntryMemoryStateSplitOperation::~LambdaEntryMemoryStateSplitOperation() noexcept = default;

bool
LambdaEntryMemoryStateSplitOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const LambdaEntryMemoryStateSplitOperation *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
LambdaEntryMemoryStateSplitOperation::debug_string() const
{
  return "LambdaEntryMemoryStateSplit";
}

std::unique_ptr<rvsdg::Operation>
LambdaEntryMemoryStateSplitOperation::copy() const
{
  return std::make_unique<LambdaEntryMemoryStateSplitOperation>(*this);
}

LambdaExitMemoryStateMergeOperation::~LambdaExitMemoryStateMergeOperation() noexcept = default;

bool
LambdaExitMemoryStateMergeOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const LambdaExitMemoryStateMergeOperation *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
LambdaExitMemoryStateMergeOperation::debug_string() const
{
  return "LambdaExitMemoryStateMerge";
}

std::unique_ptr<rvsdg::Operation>
LambdaExitMemoryStateMergeOperation::copy() const
{
  return std::make_unique<LambdaExitMemoryStateMergeOperation>(*this);
}

CallEntryMemoryStateMergeOperation::~CallEntryMemoryStateMergeOperation() noexcept = default;

bool
CallEntryMemoryStateMergeOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const CallEntryMemoryStateMergeOperation *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
CallEntryMemoryStateMergeOperation::debug_string() const
{
  return "CallEntryMemoryStateMerge";
}

std::unique_ptr<rvsdg::Operation>
CallEntryMemoryStateMergeOperation::copy() const
{
  return std::make_unique<CallEntryMemoryStateMergeOperation>(*this);
}

CallExitMemoryStateSplitOperation::~CallExitMemoryStateSplitOperation() noexcept = default;

bool
CallExitMemoryStateSplitOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const CallExitMemoryStateSplitOperation *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
CallExitMemoryStateSplitOperation::debug_string() const
{
  return "CallExitMemoryStateSplit";
}

std::unique_ptr<rvsdg::Operation>
CallExitMemoryStateSplitOperation::copy() const
{
  return std::make_unique<CallExitMemoryStateSplitOperation>(*this);
}

std::optional<std::vector<rvsdg::output *>>
NormalizeNestedMemoryStateMerges(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  std::vector<rvsdg::output *> newOperands;
  for (auto operand : operands)
  {
    auto [mergeNode, mergeOperation] = TryGetSimpleNodeAndOp<MemoryStateMergeOperation>(*operand);
    if (mergeOperation)
    {
      auto mergeOperands = rvsdg::operands(mergeNode);
      newOperands.insert(newOperands.end(), mergeOperands.begin(), mergeOperands.end());
    }
    else
    {
      newOperands.emplace_back(operand);
    }
  }

  return operands != newOperands ? newOperands : std::nullopt;
}

template<class TNode>
static std::optional<rvsdg::Node *>
UsersHaveSingleConsumer(const rvsdg::output * output)
{
  // FIXME: static assert

  if (output->IsDead())
    return std::nullopt;

  rvsdg::Node * singleConsumerNode = rvsdg::TryGetOwnerNode<TNode>(*output->begin());
  for (auto user : *output)
  {
    auto node = TryGetOwnerNode<TNode>(*output);
    if (node != singleConsumerNode)
      return std::nullopt;
  }

  return singleConsumerNode;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeMemoryStateMergeAndSplit(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  std::vector<rvsdg::output *> newOperands;
  for (auto operand : operands)
  {
    auto [splitNode, splitOperation] = TryGetSimpleNodeAndOp<MemoryStateSplitOperation>(*operand);
    if (splitOperation)
    {
      if (!UsersHaveSingleConsumer<rvsdg::SimpleNode>(operand))
        return std::nullopt;
    }
    else
    {
      newOperands.emplace_back(operand);
    }
  }
}

std::optional<std::vector<rvsdg::output *>>
NormalizeSingleOperandMemoryStateMerge(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  if (operands.size() == 1)
    return operands;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeSingleResultMemoryStateSplit(
    const MemoryStateMergeOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);

  if (operation.nresults() == 1)
    return operands;
}

}
