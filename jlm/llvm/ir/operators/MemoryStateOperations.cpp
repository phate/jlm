/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
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

std::optional<std::vector<rvsdg::Output *>>
MemoryStateMergeOperation::NormalizeSingleOperand(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  if (operands.size() == 1)
    return operands;

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
MemoryStateMergeOperation::NormalizeDuplicateOperands(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  const util::HashSet<rvsdg::Output *> uniqueOperands(operands.begin(), operands.end());

  if (uniqueOperands.Size() == operands.size())
    return std::nullopt;

  auto result = Create(std::vector(uniqueOperands.Items().begin(), uniqueOperands.Items().end()));
  return { { result } };
}

std::optional<std::vector<rvsdg::Output *>>
MemoryStateMergeOperation::NormalizeNestedMerges(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  std::vector<rvsdg::Output *> newOperands;
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

std::optional<std::vector<rvsdg::Output *>>
MemoryStateMergeOperation::NormalizeMergeSplit(
    const MemoryStateMergeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  std::vector<rvsdg::Output *> newOperands;
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

std::optional<std::vector<rvsdg::Output *>>
MemoryStateSplitOperation::NormalizeSingleResult(
    const MemoryStateSplitOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);

  if (operation.nresults() == 1)
    return operands;

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
MemoryStateSplitOperation::NormalizeNestedSplits(
    const MemoryStateSplitOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
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

std::optional<std::vector<rvsdg::Output *>>
MemoryStateSplitOperation::NormalizeSplitMerge(
    const MemoryStateSplitOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
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

std::optional<std::vector<rvsdg::Output *>>
LambdaExitMemoryStateMergeOperation::NormalizeLoad(
    const LambdaExitMemoryStateMergeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(!operands.empty());

  bool replacedOperands = false;
  std::vector<rvsdg::Output *> newOperands;
  for (auto operand : operands)
  {
    auto [loadNode, loadOperation] = rvsdg::TryGetSimpleNodeAndOp<LoadOperation>(*operand);
    if (!loadOperation)
    {
      newOperands.push_back(operand);
      continue;
    }

    auto loadAddress = LoadOperation::AddressInput(*loadNode).origin();
    auto [_, allocaOperation] = rvsdg::TryGetSimpleNodeAndOp<AllocaOperation>(*loadAddress);
    if (!allocaOperation)
    {
      newOperands.push_back(operand);
      continue;
    }

    auto newOperand = LoadOperation::MapMemoryStateOutputToInput(*operand).origin();
    newOperands.push_back(newOperand);
    replacedOperands = true;
  }

  if (!replacedOperands)
    return std::nullopt;

  return { { &Create(*operands[0]->region(), newOperands) } };
}

std::optional<std::vector<rvsdg::Output *>>
LambdaExitMemoryStateMergeOperation::NormalizeStore(
    const LambdaExitMemoryStateMergeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(!operands.empty());

  bool replacedOperands = false;
  std::vector<rvsdg::Output *> newOperands;
  for (auto operand : operands)
  {
    auto [storeNode, storeOperation] = rvsdg::TryGetSimpleNodeAndOp<StoreOperation>(*operand);
    if (!storeOperation)
    {
      newOperands.push_back(operand);
      continue;
    }

    auto storeAddress = StoreOperation::AddressInput(*storeNode).origin();
    auto [_, allocaOperation] = rvsdg::TryGetSimpleNodeAndOp<AllocaOperation>(*storeAddress);
    if (!allocaOperation)
    {
      newOperands.push_back(operand);
      continue;
    }

    auto newOperand = StoreOperation::MapMemoryStateOutputToInput(*operand).origin();
    newOperands.push_back(newOperand);
    replacedOperands = true;
  }

  if (!replacedOperands)
    return std::nullopt;

  return { { &Create(*operands[0]->region(), newOperands) } };
}

std::optional<std::vector<rvsdg::Output *>>
LambdaExitMemoryStateMergeOperation::NormalizeAlloca(
    const LambdaExitMemoryStateMergeOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(!operands.empty());

  bool replacedOperands = false;
  std::vector<rvsdg::Output *> newOperands;
  for (auto operand : operands)
  {
    auto [allocaNode, allocaOperation] = rvsdg::TryGetSimpleNodeAndOp<AllocaOperation>(*operand);
    if (allocaOperation)
    {
      auto newOperand =
          UndefValueOperation::Create(*allocaNode->region(), MemoryStateType::Create());
      newOperands.push_back(newOperand);
      replacedOperands = true;
    }
    else
    {
      newOperands.push_back(operand);
    }
  }

  if (!replacedOperands)
    return std::nullopt;

  return { { &Create(*operands[0]->region(), newOperands) } };
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
