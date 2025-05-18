/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/util/HashSet.hpp>

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

std::optional<std::vector<rvsdg::output *>>
MemoryStateMergeOperation::NormalizeSingleOperand(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  if (operands.size() == 1)
    return operands;

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
MemoryStateMergeOperation::NormalizeDuplicateStates(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  const util::HashSet<rvsdg::output *> uniqueOperands(operands.begin(), operands.end());

  if (uniqueOperands.Size() == operands.size())
    return std::nullopt;

  auto result = Create(std::vector(uniqueOperands.Items().begin(), uniqueOperands.Items().end()));
  return { { result } };
}

std::optional<std::vector<rvsdg::output *>>
MemoryStateMergeOperation::NormalizeNestedMerges(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  std::vector<rvsdg::output *> newOperands;
  for (auto operand : operands)
  {
    auto [mergeNode, mergeOperation] =
        rvsdg::TryGetSimpleNodeAndOp<MemoryStateMergeOperation>(*operand);
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

  if (operands == newOperands)
    return std::nullopt;

  auto result = Create(std::move(newOperands));
  return { { result } };
}

std::optional<std::vector<rvsdg::output *>>
MemoryStateMergeOperation::NormalizeNestedSplits(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  std::vector<rvsdg::output *> newOperands;
  for (const auto operand : operands)
  {
    auto [splitNode, splitOperation] =
        rvsdg::TryGetSimpleNodeAndOp<MemoryStateSplitOperation>(*operand);
    if (splitOperation)
    {
      newOperands.emplace_back(splitNode->input(0)->origin());
    }
    else
    {
      newOperands.emplace_back(operand);
    }
  }

  if (operands == newOperands)
    return std::nullopt;

  auto result = Create(std::move(newOperands));
  return { { result } };
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

std::optional<std::vector<rvsdg::output *>>
MemoryStateSplitOperation::NormalizeSingleResult(
    const MemoryStateSplitOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);

  if (operation.nresults() == 1)
    return operands;

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
MemoryStateSplitOperation::NormalizeNestedSplits(
    const MemoryStateSplitOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  const auto operand = operands[0];

  auto [splitNode, splitOperation] =
      rvsdg::TryGetSimpleNodeAndOp<MemoryStateSplitOperation>(*operand);
  if (!splitOperation)
    return std::nullopt;

  const auto numResults = splitOperation->nresults() + operation.nresults();
  auto & newOperand = *splitNode->input(0)->origin();
  auto results = Create(newOperand, numResults);

  for (size_t n = 0; n < splitNode->noutputs(); n++)
  {
    const auto output = splitNode->output(n);
    output->divert_users(results[n]);
  }

  return { { std::next(results.begin(), splitNode->noutputs()), results.end() } };
}

std::optional<std::vector<rvsdg::output *>>
MemoryStateSplitOperation::NormalizeSplitMerge(
    const MemoryStateSplitOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  const auto operand = operands[0];

  auto [mergeNode, mergeOperation] =
      rvsdg::TryGetSimpleNodeAndOp<MemoryStateMergeOperation>(*operand);
  if (!mergeOperation || mergeOperation->narguments() != operation.nresults())
    return std::nullopt;

  return { rvsdg::operands(mergeNode) };
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

}
