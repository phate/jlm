/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/opt/LoadChainSeparation.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <RVSDG/Ops.h.inc>

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

  util::HashSet<rvsdg::Output *> loadChainEnds;
  findLoadChainEnds(module.Rvsdg().GetRootRegion(), loadChainEnds);

  for (auto & memoryStateOutput : loadChainEnds.Items())
  {
    separateLoadChain(*memoryStateOutput);
  }
}

void
LoadChainSeparation::findLoadChainEnds(
    rvsdg::Region & region,
    util::HashSet<rvsdg::Output *> & loadChainEnds)
{
  for (auto & node : region.Nodes())
  {
    // Handle innermost regions first
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        findLoadChainEnds(subregion, loadChainEnds);
      }
    }

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
        loadChainEnds.insert(&memoryStateOutput);
      }
    }
  }
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
  const auto & operand = *input.origin();

  if (rvsdg::IsOwnerNodeOperation<LoadOperation>(operand))
    return true;

  // Handle gamma outputs
  if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(operand))
  {
    auto [branchResults, _] = gammaNode->MapOutputExitVar(operand);
    for (const auto branchResult : branchResults)
    {
      if (hasLoadNodeAsOperandOwner(*branchResult))
        return true;
    }

    return false;
  }

  // Handle theta outputs
  if (const auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(operand))
  {
    const auto loopVar = thetaNode->MapOutputLoopVar(operand);
    return hasLoadNodeAsOperandOwner(*loopVar.post);
  }

  // Handle gamma subregion arguments
  if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(operand))
  {
    const auto roleVar = gammaNode->MapBranchArgument(operand);
    if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
    {
      return hasLoadNodeAsOperandOwner(*entryVar->input);
    }

    return false;
  }

  // Handle theta subregion arguments
  if (const auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(operand))
  {
    const auto loopVar = thetaNode->MapOutputLoopVar(operand);
    return hasLoadNodeAsOperandOwner(*loopVar.input);
  }

  return false;
}

bool
LoadChainSeparation::hasLoadNodeAsUserOwner(const rvsdg::Output & output)
{
  for (auto & user : output.Users())
  {
    if (rvsdg::IsOwnerNodeOperation<LoadOperation>(user))
      return true;

    // Handle gamma inputs
    if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user))
    {
      const auto roleVar = gammaNode->MapInput(user);
      if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
      {
        for (const auto branchArgument : entryVar->branchArgument)
        {
          if (hasLoadNodeAsUserOwner(*branchArgument))
          {
            return true;
          }
        }
      }

      return false;
    }

    // Handle theta inputs
    if (const auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(user))
    {
      const auto loopVar = thetaNode->MapInputLoopVar(user);
      return hasLoadNodeAsUserOwner(*loopVar.pre);
    }

    // Handle gamma subregion results
    if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(user))
    {
      const auto [_, gammaOutput] = gammaNode->MapBranchResultExitVar(user);
      return hasLoadNodeAsUserOwner(*gammaOutput);
    }

    // Handle theta subregion results
    if (const auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(user))
    {
      const auto loopVar = thetaNode->MapPostLoopVar(user);
      return hasLoadNodeAsUserOwner(*loopVar.output);
    }
  }

  return false;
}

}
