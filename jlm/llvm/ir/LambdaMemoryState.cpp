/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>

namespace jlm::llvm
{

rvsdg::output &
GetMemoryStateRegionArgument(const rvsdg::LambdaNode & lambdaNode) noexcept
{
  auto argument = lambdaNode.GetFunctionArguments().back();
  JLM_ASSERT(is<MemoryStateType>(argument->type()));
  return *argument;
}

rvsdg::input &
GetMemoryStateRegionResult(const rvsdg::LambdaNode & lambdaNode) noexcept
{
  auto result = lambdaNode.GetFunctionResults().back();
  JLM_ASSERT(is<MemoryStateType>(result->type()));
  return *result;
}

rvsdg::SimpleNode *
GetMemoryStateExitMerge(const rvsdg::LambdaNode & lambdaNode) noexcept
{
  auto & result = GetMemoryStateRegionResult(lambdaNode);

  auto node = rvsdg::output::GetNode(*result.origin());
  return is<LambdaExitMemoryStateMergeOperation>(node) ? dynamic_cast<rvsdg::SimpleNode *>(node)
                                                       : nullptr;
}

rvsdg::SimpleNode *
GetMemoryStateEntrySplit(const rvsdg::LambdaNode & lambdaNode) noexcept
{
  auto & argument = GetMemoryStateRegionArgument(lambdaNode);

  // If a memory state entry split node is present, then we would expect the node to be the only
  // user of the memory state argument.
  if (argument.nusers() != 1)
    return nullptr;

  auto node = rvsdg::node_input::GetNode(**argument.begin());
  return is<LambdaEntryMemoryStateSplitOperation>(node) ? dynamic_cast<rvsdg::SimpleNode *>(node)
                                                        : nullptr;
}

}
