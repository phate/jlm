/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <algorithm>
#include <jlm/hls/backend/rvsdg2rhls/DeadNodeElimination.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static bool
RemoveUnusedLoopOutputs(LoopNode & loopNode)
{
  // Keep only those entry vars that are not dead.
  std::vector<LoopNode::ExitVar> vars = loopNode.getExitVars();
  vars.erase(
      std::remove_if(
          vars.begin(),
          vars.end(),
          [](const LoopNode::ExitVar & var)
          {
            return !var.output->IsDead();
          }),
      vars.end());

  // Remove all dead vars.
  bool anyChanged = !vars.empty();
  loopNode.removeExitVars(std::move(vars));
  return anyChanged;
}

static bool
RemoveUnusedInputs(LoopNode & loopNode)
{
  // Keep only those entry vars that are not dead.
  std::vector<LoopNode::EntryVar> vars = loopNode.getEntryVars();
  vars.erase(
      std::remove_if(
          vars.begin(),
          vars.end(),
          [](const LoopNode::EntryVar & var)
          {
            return !var.inner->IsDead();
          }),
      vars.end());

  // Remove all dead vars.
  bool anyChanged = !vars.empty();
  loopNode.removeEntryVars(std::move(vars));
  return anyChanged;
}

static bool
RemoveUnusedBackEdges(LoopNode & loopNode)
{
  // Keep only back edge vars that have a user (instead of
  // simply forwarding to itself).
  std::vector<LoopNode::BackEdgeVar> vars = loopNode.getBackEdgeVars();
  vars.erase(
      std::remove_if(
          vars.begin(),
          vars.end(),
          [](const LoopNode::BackEdgeVar & var)
          {
            return !(var.pre->nusers() == 1 && var.post->origin() == var.pre);
          }),
      vars.end());
  // Remove all that have exactly one user, namely forward itself
  // to next loop iteration.
  bool anyChanged = !vars.empty();
  loopNode.removeBackEdgeVars(std::move(vars));
  return anyChanged;
}

static bool
EliminateDeadNodesInRegion(rvsdg::Region & region)
{
  bool changed = false;
  bool anyChanged = false;

  do
  {
    changed = false;
    for (auto & node : rvsdg::BottomUpTraverser(&region))
    {
      if (node->IsDead())
      {
        remove(node);
        changed = true;
      }
      else if (auto loopNode = dynamic_cast<LoopNode *>(node))
      {
        changed |= RemoveUnusedLoopOutputs(*loopNode);
        changed |= RemoveUnusedInputs(*loopNode);
        changed |= RemoveUnusedBackEdges(*loopNode);
        changed |= EliminateDeadNodesInRegion(*loopNode->subregion());
      }
    }
    anyChanged |= changed;
  } while (changed);

  JLM_ASSERT(region.numBottomNodes() == 0);
  return anyChanged;
}

void
EliminateDeadNodes(llvm::LlvmRvsdgModule & rvsdgModule)
{
  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();

  if (rootRegion.numNodes() != 1)
  {
    throw util::Error("Root should have only one node now");
  }

  auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(rootRegion.Nodes().begin().ptr());
  if (!lambdaNode)
  {
    throw util::Error("Node needs to be a lambda");
  }

  EliminateDeadNodesInRegion(*lambdaNode->subregion());
}

}
