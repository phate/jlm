/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>

namespace jlm::llvm
{

rvsdg::Output &
GetMemoryStateRegionArgument(const rvsdg::LambdaNode & lambdaNode) noexcept
{
  auto argument = lambdaNode.GetFunctionArguments().back();
  JLM_ASSERT(is<MemoryStateType>(argument->Type()));
  return *argument;
}

rvsdg::Input &
GetMemoryStateRegionResult(const rvsdg::LambdaNode & lambdaNode) noexcept
{
  auto result = lambdaNode.GetFunctionResults().back();
  JLM_ASSERT(is<MemoryStateType>(result->Type()));
  return *result;
}

rvsdg::SimpleNode *
GetMemoryStateExitMerge(const rvsdg::LambdaNode & lambdaNode) noexcept
{
  auto & result = GetMemoryStateRegionResult(lambdaNode);

  const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*result.origin());
  return is<LambdaExitMemoryStateMergeOperation>(node) ? node : nullptr;
}

rvsdg::SimpleNode *
GetMemoryStateEntrySplit(const rvsdg::LambdaNode & lambdaNode) noexcept
{
  auto & argument = GetMemoryStateRegionArgument(lambdaNode);

  // If a memory state entry split node is present, then we would expect the node to be the only
  // user of the memory state argument.
  if (argument.nusers() != 1)
    return nullptr;

  const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(argument.SingleUser());
  return is<LambdaEntryMemoryStateSplitOperation>(node) ? dynamic_cast<rvsdg::SimpleNode *>(node)
                                                        : nullptr;
}

}
