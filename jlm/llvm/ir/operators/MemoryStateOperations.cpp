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
        rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(*operand);
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
        rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateSplitOperation>(*operand);
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
      rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateSplitOperation>(*operand);
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
      rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(*operand);
  if (!mergeOperation || mergeOperation->narguments() != operation.nresults())
    return std::nullopt;

  return { rvsdg::operands(mergeNode) };
}

static void
CheckMemoryNodeIds(
    const std::vector<MemoryNodeId> & memoryNodeIds,
    const size_t numExpectedMemoryNodeIds)
{
  if (memoryNodeIds.size() != numExpectedMemoryNodeIds)
    throw std::logic_error("Insufficient number of memory node identifiers");

  const util::HashSet<MemoryNodeId> memoryNodeIdsSet(
      { memoryNodeIds.begin(), memoryNodeIds.end() });

  if (memoryNodeIdsSet.Size() != numExpectedMemoryNodeIds)
    throw std::logic_error("Found duplicated memory node identifiers.");
}

static std::string
ToString(const std::vector<MemoryNodeId> & memoryNodeIds)
{
  std::string str;
  for (size_t n = 0; n < memoryNodeIds.size(); n++)
  {
    str.append(util::strfmt(memoryNodeIds[n]));
    if (n != memoryNodeIds.size() - 1)
      str.append(", ");
  }

  return str;
}

LambdaEntryMemoryStateSplitOperation::LambdaEntryMemoryStateSplitOperation(
    const size_t numResults,
    std::vector<MemoryNodeId> memoryNodeIds)
    : MemoryStateOperation(1, numResults),
      MemoryNodeIds_(std::move(memoryNodeIds))
{
  CheckMemoryNodeIds(MemoryNodeIds_, numResults);
}

LambdaEntryMemoryStateSplitOperation::~LambdaEntryMemoryStateSplitOperation() noexcept = default;

bool
LambdaEntryMemoryStateSplitOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const LambdaEntryMemoryStateSplitOperation *>(&other);
  return operation && operation->nresults() == nresults()
      && operation->MemoryNodeIds_ == MemoryNodeIds_;
}

std::string
LambdaEntryMemoryStateSplitOperation::debug_string() const
{
  return util::strfmt("LambdaEntryMemoryStateSplit[", ToString(MemoryNodeIds_), "]");
}

std::unique_ptr<rvsdg::Operation>
LambdaEntryMemoryStateSplitOperation::copy() const
{
  return std::make_unique<LambdaEntryMemoryStateSplitOperation>(*this);
}

LambdaExitMemoryStateMergeOperation::LambdaExitMemoryStateMergeOperation(
    std::vector<MemoryNodeId> memoryNodeIds)
    : MemoryStateOperation(memoryNodeIds.size(), 1),
      MemoryNodeIds_(std::move(memoryNodeIds))
{
  CheckMemoryNodeIds(MemoryNodeIds_, MemoryNodeIds_.size());
}

LambdaExitMemoryStateMergeOperation::~LambdaExitMemoryStateMergeOperation() noexcept = default;

bool
LambdaExitMemoryStateMergeOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const LambdaExitMemoryStateMergeOperation *>(&other);
  return operation && operation->MemoryNodeIds_ == MemoryNodeIds_;
}

std::string
LambdaExitMemoryStateMergeOperation::debug_string() const
{
  return util::strfmt("LambdaExitMemoryStateMerge[", ToString(MemoryNodeIds_), "]");
}

std::unique_ptr<rvsdg::Operation>
LambdaExitMemoryStateMergeOperation::copy() const
{
  return std::make_unique<LambdaExitMemoryStateMergeOperation>(*this);
}

std::optional<std::vector<rvsdg::Output *>>
LambdaExitMemoryStateMergeOperation::NormalizeLoadFromAlloca(
    const LambdaExitMemoryStateMergeOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (operands.empty())
    return std::nullopt;

  bool replacedOperands = false;
  std::vector<rvsdg::Output *> newOperands;
  for (auto operand : operands)
  {
    auto [loadNode, loadOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<LoadOperation>(*operand);
    if (!loadOperation)
    {
      newOperands.push_back(operand);
      continue;
    }

    auto loadAddress = LoadOperation::AddressInput(*loadNode).origin();
    auto [_, allocaOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(*loadAddress);
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

  return {
    { CreateNode(*operands[0]->region(), newOperands, operation.MemoryNodeIds_).output(0) }
  };
}

std::optional<std::vector<rvsdg::Output *>>
LambdaExitMemoryStateMergeOperation::NormalizeStoreToAlloca(
    const LambdaExitMemoryStateMergeOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (operands.empty())
    return std::nullopt;

  bool replacedOperands = false;
  std::vector<rvsdg::Output *> newOperands;
  for (auto operand : operands)
  {
    auto [storeNode, storeOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<StoreOperation>(*operand);
    if (!storeOperation)
    {
      newOperands.push_back(operand);
      continue;
    }

    auto storeAddress = StoreOperation::AddressInput(*storeNode).origin();
    auto [_, allocaOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(*storeAddress);
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

  return {
    { CreateNode(*operands[0]->region(), newOperands, operation.GetMemoryNodeIds()).output(0) }
  };
}

std::optional<std::vector<rvsdg::Output *>>
LambdaExitMemoryStateMergeOperation::NormalizeAlloca(
    const LambdaExitMemoryStateMergeOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (operands.empty())
    return std::nullopt;

  bool replacedOperands = false;
  std::vector<rvsdg::Output *> newOperands;
  for (auto operand : operands)
  {
    auto [allocaNode, allocaOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(*operand);
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

  return {
    { CreateNode(*operands[0]->region(), newOperands, operation.GetMemoryNodeIds()).output(0) }
  };
}

CallEntryMemoryStateMergeOperation::~CallEntryMemoryStateMergeOperation() noexcept = default;

CallEntryMemoryStateMergeOperation::CallEntryMemoryStateMergeOperation(
    std::vector<MemoryNodeId> memoryNodeIds)
    : MemoryStateOperation(memoryNodeIds.size(), 1),
      MemoryNodeIds_(std::move(memoryNodeIds))
{
  CheckMemoryNodeIds(MemoryNodeIds_, MemoryNodeIds_.size());
}

bool
CallEntryMemoryStateMergeOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const CallEntryMemoryStateMergeOperation *>(&other);
  return operation && operation->MemoryNodeIds_ == MemoryNodeIds_;
}

std::string
CallEntryMemoryStateMergeOperation::debug_string() const
{
  return util::strfmt("CallEntryMemoryStateMerge[", ToString(MemoryNodeIds_), "]");
}

std::unique_ptr<rvsdg::Operation>
CallEntryMemoryStateMergeOperation::copy() const
{
  return std::make_unique<CallEntryMemoryStateMergeOperation>(*this);
}

CallExitMemoryStateSplitOperation::~CallExitMemoryStateSplitOperation() noexcept = default;

CallExitMemoryStateSplitOperation::CallExitMemoryStateSplitOperation(
    std::vector<MemoryNodeId> memoryNodeIds)
    : MemoryStateOperation(1, memoryNodeIds.size()),
      MemoryNodeIds_(std::move(memoryNodeIds))
{
  CheckMemoryNodeIds(MemoryNodeIds_, MemoryNodeIds_.size());
}

bool
CallExitMemoryStateSplitOperation::operator==(const Operation & other) const noexcept
{
  const auto operation = dynamic_cast<const CallExitMemoryStateSplitOperation *>(&other);
  return operation && operation->MemoryNodeIds_ == MemoryNodeIds_;
}

std::string
CallExitMemoryStateSplitOperation::debug_string() const
{
  return util::strfmt("CallExitMemoryStateSplit[", ToString(MemoryNodeIds_), "]");
}

std::unique_ptr<rvsdg::Operation>
CallExitMemoryStateSplitOperation::copy() const
{
  return std::make_unique<CallExitMemoryStateSplitOperation>(*this);
}

std::optional<std::vector<rvsdg::Output *>>
CallExitMemoryStateSplitOperation::NormalizeLambdaExitMemoryStateMerge(
    const CallExitMemoryStateSplitOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);

  auto [lambdaExitMergeNode, lambdaExitMergeOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<LambdaExitMemoryStateMergeOperation>(*operands[0]);
  if (!lambdaExitMergeOperation)
    return std::nullopt;

  JLM_ASSERT(lambdaExitMergeNode->ninputs() == operation.nresults());
  return rvsdg::operands(lambdaExitMergeNode);
}

}
