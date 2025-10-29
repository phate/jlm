/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/opt/LoadChainSeparation.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/structural-node.hpp>

namespace jlm::llvm
{
LoadChainSeparation::~LoadChainSeparation() noexcept = default;

LoadChainSeparation::LoadChainSeparation()
    : Transformation("LoadChainSeparation")
{}

void
LoadChainSeparation::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  handleRegion(module.Rvsdg().GetRootRegion());
}

void
LoadChainSeparation::handleRegion(rvsdg::Region & region)
{
  // Handle innermost regions first
  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        handleRegion(subregion);
      }
    }
  }

  // Separate load chains
  const auto loadChainBottoms = findLoadChainBottoms(region);
  for (auto & memoryStateOutput : loadChainBottoms.Items())
  {
    separateLoadChain(*memoryStateOutput);
  }
}

void
LoadChainSeparation::separateLoadChain(rvsdg::Output & memoryStateOutput)
{
  JLM_ASSERT(rvsdg::is<MemoryStateType>(memoryStateOutput.Type()));
  JLM_ASSERT(rvsdg::IsOwnerNodeOperation<LoadOperation>(memoryStateOutput));

  std::vector<rvsdg::Output *> joinOperands;
  auto & newMemoryStateOperand = traceLoadNodeMemoryState(memoryStateOutput, joinOperands);
  JLM_ASSERT(joinOperands.size() > 1);

  // Divert the operands of the respective inputs for each encountered memory state output
  for (const auto output : joinOperands)
  {
    auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(*output);
    memoryStateInput.divert_to(&newMemoryStateOperand);
  }

  // Create join node and divert the current memory state output
  auto & joinNode = MemoryStateJoinOperation::CreateNode(joinOperands);
  memoryStateOutput.divertUsersWhere(
      *joinNode.output(0),
      [&joinNode](const rvsdg::Input & user)
      {
        return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user) != &joinNode;
      });
}

rvsdg::Output &
LoadChainSeparation::traceLoadNodeMemoryState(
    rvsdg::Output & output,
    std::vector<rvsdg::Output *> & joinOperands)
{
  JLM_ASSERT(rvsdg::is<MemoryStateType>(output.Type()));

  if (!is<LoadOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output)))
    return output;

  joinOperands.push_back(&output);
  return traceLoadNodeMemoryState(
      *LoadOperation::MapMemoryStateOutputToInput(output).origin(),
      joinOperands);
}

util::HashSet<rvsdg::Output *>
LoadChainSeparation::findLoadChainBottoms(rvsdg::Region & region)
{
  util::HashSet<rvsdg::Output *> loadChainBottoms;
  for (auto & node : region.Nodes())
  {
    if (!rvsdg::is<LoadOperation>(&node))
    {
      continue;
    }

    for (auto & memoryStateOutput : LoadOperation::MemoryStateOutputs(node))
    {
      if (hasLoadNodeAsUserOwner(memoryStateOutput))
      {
        continue;
      }

      auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(memoryStateOutput);
      if (hasLoadNodeAsOperandOwner(memoryStateInput))
      {
        loadChainBottoms.insert(&memoryStateOutput);
      }
    }
  }

  return loadChainBottoms;
}

bool
LoadChainSeparation::hasLoadNodeAsOperandOwner(const rvsdg::Input & input)
{
  return rvsdg::IsOwnerNodeOperation<LoadOperation>(*input.origin());
}

bool
LoadChainSeparation::hasLoadNodeAsUserOwner(const rvsdg::Output & output)
{
  for (auto & user : output.Users())
  {
    if (rvsdg::IsOwnerNodeOperation<LoadOperation>(user))
      return true;
  }

  return false;
}

}
