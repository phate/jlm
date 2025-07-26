/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/DeadNodeElimination.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static bool
RemoveUnusedLoopOutputs(LoopNode & loopNode)
{
  bool anyChanged = false;
  auto loopSubregion = loopNode.subregion();

  // go through in reverse because we might remove outputs
  for (int i = loopNode.noutputs() - 1; i >= 0; --i)
  {
    auto output = loopNode.output(i);

    if (output->nusers() == 0)
    {
      JLM_ASSERT(output->results.size() == 1);
      auto result = output->results.begin();
      loopSubregion->RemoveResult(result->index());
      loopNode.RemoveOutput(output->index());
      anyChanged = true;
    }
  }
  return anyChanged;
}

static bool
RemoveUnusedInputs(LoopNode & loopNode)
{
  bool anyChanged = false;
  auto loopSubregion = loopNode.subregion();

  // go through in reverse because we might remove inputs
  for (int i = loopNode.ninputs() - 1; i >= 0; --i)
  {
    auto input = loopNode.input(i);
    JLM_ASSERT(input->arguments.size() == 1);
    auto argument = input->arguments.begin();

    if (argument->nusers() == 0)
    {
      loopSubregion->RemoveArgument(argument->index());
      loopNode.RemoveInput(input->index());
      anyChanged = true;
    }
  }

  // clean up unused arguments - only ones without an input should be left
  // go through in reverse because we might remove some
  for (int i = loopSubregion->narguments() - 1; i >= 0; --i)
  {
    auto argument = loopSubregion->argument(i);

    if (auto backedgeArgument = dynamic_cast<backedge_argument *>(argument))
    {
      auto result = backedgeArgument->result();
      JLM_ASSERT(*result->Type() == *argument->Type());

      if (argument->nusers() == 0 || (argument->nusers() == 1 && result->origin() == argument))
      {
        loopSubregion->RemoveResult(result->index());
        loopSubregion->RemoveArgument(argument->index());
      }
    }
    else
    {
      JLM_ASSERT(argument->nusers() != 0);
    }
  }

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
        changed |= EliminateDeadNodesInRegion(*loopNode->subregion());
      }
    }
    anyChanged |= changed;
  } while (changed);

  JLM_ASSERT(region.NumBottomNodes() == 0);
  return anyChanged;
}

void
EliminateDeadNodes(llvm::RvsdgModule & rvsdgModule)
{
  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();

  if (rootRegion.nnodes() != 1)
  {
    throw util::error("Root should have only one node now");
  }

  auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(rootRegion.Nodes().begin().ptr());
  if (!lambdaNode)
  {
    throw util::error("Node needs to be a lambda");
  }

  EliminateDeadNodesInRegion(*lambdaNode->subregion());
}

}
